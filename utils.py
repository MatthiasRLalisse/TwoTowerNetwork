import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch import nn
import gc

class defaultTrueDict(dict):
    def __missing__(self,key):
        return True

def get_embedding(term, embeddings):
    # Get embedding for query
    embed_dim = next(iter(embeddings.values())).shape[0]
    try:
        return embeddings[term]
    except KeyError:
        print (f"Missing query: {term}")
        return torch.zeros(embed_dim, dtype=torch.float32) #.to(device)

class Evaluator(object):
  """Utility class to compute rankings from model predictions."""
  def __init__(self, data, query_embeddings, 
                     title_embeddings, sim_metric=nn.CosineSimilarity ):
    ###data should be an array of size (n_samples x 2) where the
    ###	first column is titles and the second column is queries.
    ###pre-compiles a filter to remove true keywords from the rankings.
    ###given an array of similarity scores, returns the rank of the true 
    ###	instance in the array.
    
    #_filter_dict returns False if (title, query) is in the dataset, 
    #	else True. This filters out true instances in the ranking metrics
    self.data = data[:]
    self._filter_dict = defaultTrueDict()
    self.t2neighbors, self.q2neighbors = defaultdict(set), defaultdict(set)
    for title, query in self.data:
      self._filter_dict[(title, query)] = False
      self.t2neighbors[title].add(query)
      self.q2neighbors[query].add(title)
    
    self._all_queries = np.array(list(query_embeddings.keys()))
    self._all_titles = np.array(list(title_embeddings.keys()))
    self.query2idx = { q: i for i, q in enumerate(self._all_queries) }
    self.title2idx = { t: i for i, t in enumerate(self._all_titles) }
    
    self.q2ignore_idx = defaultdict(set)
    self.t2ignore_idx = defaultdict(set)
    for title, query in self.data:
      self.q2ignore_idx[query].add(self.title2idx[title])
      self.t2ignore_idx[title].add(self.query2idx[query])
    
    self.t2ignore_idx = defaultdict(list, { q: list(st) for q, st in \
                                               self.t2ignore_idx.items() })
    self.q2ignore_idx = defaultdict(list, { t: list(st) for t, st in \
                                               self.q2ignore_idx.items() })
    
    
    self.q_embed = query_embeddings
    self.t_embed = title_embeddings
    
    self.sim_metric = sim_metric()
    
  def _get_filter(self, input_, _filter_dict=None):
    _filter_dict = self._filter_dict if _filter_dict is None else _filter_dict
    filt = np.array([ _filter_dict[(title, query)] for title, query in input_ ])
    return filt
  
  def _filter(self, input_, _filter_dict=None):
    ### filters an input set of queries
    return np.array(input_)[self._get_filter(input_, _filter_dict)]
  
  def _get_candidates(self, term, queries=True, _filter_dict=None):
    _filter_dict = self._filter_dict if _filter_dict is None else self._filter_dict
    if queries:
      candidates = self._filter([ (term, query) for query in self._all_queries ], _filter_dict)
    else:
      #if queries is false, rank the titles with 
      #	respect to the term (which is a query)
      candidates = self._filter([ (title, term) for title in self._all_titles ], _filter_dict)
    return candidates
  
  def get_rank(self, term, true_response, model, 
                     queries=True, return_ranks=False, 
                     batch_size=200_000):
    ### "queries" kwarg determines whether the rank is of 
    ###	the queries w.r.t. to the title or vice versa
    
    ### SLOW METHOD. See eval_ function for a much faster implementation
    ###	that pre-compiles embeddings for the entire test set.
    candidates = self._get_candidates(term, queries=queries)
    device = next(model.parameters()).device
    if queries:
      input_queries = torch.stack([ get_embedding(true_response, self.q_embed) ] \
                                    + [ get_embedding(query, self.q_embed)  \
                                    for query in candidates[:,1] ]).to(device)
      #input_titles = torch.stack([ get_embedding(term, self.t_embed) \
      #                                 ]*(len(candidates)+1) ).to(device)
      input_titles = torch.stack([ get_embedding(term, self.t_embed) ]).to(device)
    else:
      input_titles = torch.stack([ get_embedding(true_response, self.t_embed) ] \
                                    + [ get_embedding(title, self.t_embed)   \
                                    for title in candidates[:,0] ]).to(device)
      #input_queries = torch.stack([ get_embedding(term, self.q_embed)  \
      #                                 ]*(len(candidates)+1) ).to(device)
      input_queries = torch.stack([ get_embedding(term, self.q_embed) ]).to(device)
    
    with torch.no_grad():
      all_similarities = []
      for batch in range(0, len(candidates)+1, batch_size):
        batch_titles = input_titles[batch:batch+batch_size] \
                             if not queries else input_titles
        batch_queries = input_queries[batch:batch+batch_size] \
                             if queries else input_queries
        similarities = self.sim_metric(*model(batch_titles, batch_queries))
        all_similarities.append(similarities.detach().cpu().numpy())
      
      all_similarities = np.concatenate(all_similarities)
    
    ranks = (-all_similarities).argsort().argsort() + 1
    
    #first entry is the true response
    if return_ranks:
      return ranks[0], ranks
    else:
      return ranks[0]
  
  def _compile_output_embeds(self, model, batch_size=50_000, 
                             titles=None, queries=None):
    device = next(model.parameters()).device
    q_outputs = {}
    queries = self._all_queries if queries is None else np.array(queries)
    for batch in range(0, len(queries)+1, batch_size):
      query_terms = queries[batch:batch+batch_size]
      input_queries = torch.stack([ get_embedding(q, self.q_embed) 
                                             for q in query_terms ])
      with torch.no_grad():
        output_queries = model.query_convert(input_queries.to(device)).detach().cpu()
      
      q_outputs.update({ q: q_embed for q, q_embed in 
                            zip(query_terms, output_queries) })
    
    titles = self._all_titles if titles is None else np.array(titles)
    t_outputs = {}
    for batch in range(0, len(titles)+1, batch_size):
      title_terms = titles[batch:batch+batch_size]
      input_titles = torch.stack([ get_embedding(t, self.t_embed) 
                                             for t in title_terms ])
      with torch.no_grad():
        output_titles = model.title_convert(input_titles.to(device)).detach().cpu()
      
      t_outputs.update({ t: t_embed for t, t_embed in 
                            zip(title_terms, output_titles) })
    
    return t_outputs, q_outputs
  
  def _compile_cand_ids4subset(self, title_samp, query_samp):
    
    __title2id = { t: i for i, t in enumerate(title_samp) }
    __query2id = { q: i for i, q in enumerate(query_samp) }
    
    title2ids = {} #title: np.vstack([np.array(title_samp), np.array(query_samp)).T
    for title in tqdm(title_samp, desc='compiling candidates (queries)',
                                  positon=0, leave=True):
      in_sample_candidate_ids = [ __query2id[q] for q in query_samp if q not in self.t2neighbors[title] ]
      #for _, cand in self._get_candidates(title, queries=True):
      #  #print(cand)
      #  try: in_sample_candidate_ids.append(__query2id[cand])
      #  except KeyError: pass
      title2ids[title] = np.array(in_sample_candidate_ids)
    
    query2ids = {}
    for query in tqdm(query_samp, desc='compiling candidates (titles)', 
                                  position=0, leave=True):
      in_sample_candidate_ids = [ __title2id[t] for t in title_samp if t not in self.q2neighbors[query] ]
      #for cand, _ in self._get_candidates(query, queries=False):
      #  try: in_sample_candidate_ids.append(__title2id[cand])
      #  except KeyError: pass
      query2ids[query] = np.array(in_sample_candidate_ids)
    
    return title2ids, query2ids
  
  def eval(self, test_data, model, precomp_batch_size=50_000, 
                                   sim_batch_size=200_000, 
                                   filter_outputs=True, 
                                   eval_sample_size=None):
    ### FAST evaluation on the entire dataset
    #begin by gathering model-generated embeddings for all
    #	titles and queries
    #	if eval_sample_size is not None, only 
    #		sample a subset of titles/queries
    #		which speeds up train/validation loop
    device = next(model.parameters()).device
    
    if eval_sample_size is not None:
      print('compiling candidate set for eval')
      incl_titles = set(test_data[:,0])
      incl_queries = set(test_data[:,1])
      
      print(len(incl_titles), len(incl_queries))
      
      title_samp_size = max(eval_sample_size, len(incl_titles))
      query_samp_size = max(eval_sample_size, len(incl_queries))
      
      pbar = tqdm(total=title_samp_size-len(incl_titles), 
                  desc='titles', position=0, leave=True)
      pbar.set_postfix({'n': f'{len(incl_titles)}'})
      i = 0; title_perm = np.random.permutation(self._all_titles)
      while len(incl_titles) < title_samp_size and i < len(self._all_titles):	
        incl_titles.add(title_perm[i]); i += 1
        #print('\n\n\n')
        _ = pbar.update(); 
        #print(i, '\n\n\n')
        pbar.set_postfix({'n': f'{len(incl_titles)}'})
        #print('titles', i, len(incl_titles))
      incl_titles = list(incl_titles)
      
      #del pbar; _ = gc.collect()
      pbar2 = tqdm(total=query_samp_size-len(incl_queries), 
                   desc='titles', position=0, leave=True)
      pbar2.set_postfix({'n': f'{len(incl_queries)}'})
      i = 0; query_perm = np.random.permutation(self._all_queries)
      while len(incl_queries) < query_samp_size and i < len(self._all_queries):
        incl_queries.add(query_perm[i]); i += 1
        _ = pbar2.update()
        pbar2.set_postfix({'n': f'{len(incl_queries)}'})
        #print('queries', i, len(incl_queries))
      incl_queries = list(incl_queries)
      
      print('compiling candidates for the subset')
      t2cand_ids, q2cand_ids = self._compile_cand_ids4subset(
                                   incl_titles, incl_queries)
    else: 
      incl_queries = self._all_queries
      incl_titles = self._all_titles
    
    print('compiling embeddings')
    t_outputs, q_outputs = self._compile_output_embeds(model, 
                                 batch_size=precomp_batch_size, 
                                 titles=incl_titles, 
                                 queries=incl_queries)
    t_embedz, q_embedz = torch.stack(list(t_outputs.values())), \
                         torch.stack(list(q_outputs.values()))
    
    #if filter_outputs:
    #  t_embedz, q_embedz = t_embedz.detach().cpu().numpy(), \
    #                       q_embedz.detach().cpu().numpy()
    
    #iterate across test data and get ranks
    t_ranks, q_ranks = [], []
    mean_mrr_q = mean_mrr_t = 0
    #progress_str = f'query mrr: {mean_mrr_q:.4f}, title mrr: {mean_mrr_t:.4f}'
    pbar = tqdm(test_data, position=0, leave=True) #, desc=progress_str)
    pbar.set_postfix({'query_mrr': '0', 
                       'title_mrr': '0' })
    for title, query in pbar:
      title_id, query_id = self.title2idx[title], self.query2idx[query]
      t_embedding = torch.stack([get_embedding(title, t_outputs)])
      q_embedding = torch.stack([get_embedding(query, q_outputs)])
      
      #print(t_embedding, q_embedding)
      #rank the queries
      if eval_sample_size is None:
        q_neg_examples = q_embedz
      else:
        q_neg_examples = q_embedz[t2cand_ids[title]]
      q_output_titles = t_embedding #torch.stack([ t_embedding ])
      q_output_queries = torch.cat([q_embedding, q_neg_examples ], axis=0)
      
      q_all_similarities = []
      for batch in range(0, q_output_queries.shape[0], sim_batch_size):
       with torch.no_grad():
        similarities = self.sim_metric(q_output_titles.to(device), 
                            q_output_queries[batch:batch+sim_batch_size].to(device))
        q_all_similarities.append(similarities.detach().cpu().numpy())
      
      q_all_similarities = np.concatenate(q_all_similarities)
      q_ranks_ = (-q_all_similarities).argsort().argsort() + 1
      
      #rank the titles
      t_output_queries = q_embedding
      if eval_sample_size is None:
        t_neg_examples = t_embedz
      else:
        t_neg_examples = t_embedz[q2cand_ids[query]]
      #t_neg_examples = t_embedz
      q_output_titles = t_embedding #torch.stack([ t_embedding ])
      
      t_output_titles = torch.cat([t_embedding, t_neg_examples ], axis=0)
      
      t_all_similarities = []
      for batch in range(0, t_output_titles.shape[0], sim_batch_size):
       with torch.no_grad():
        similarities = self.sim_metric(t_output_titles[batch: \
                            batch+sim_batch_size].to(device), 
                            t_output_queries.to(device))
        t_all_similarities.append(similarities.detach().cpu().numpy())
      
      t_all_similarities = np.concatenate(t_all_similarities)
      t_ranks_ = (-t_all_similarities).argsort().argsort() + 1
      
      #print('t', t_all_similarities)
      #print(t_ranks_)
      #print('q', q_all_similarities)
      #print(q_ranks_)
      
      q_ranks.append(q_ranks_[0]); t_ranks.append(t_ranks_[0])
      mean_mrr_q = np.mean(1/np.array(q_ranks))
      mean_mrr_t = np.mean(1/np.array(t_ranks))
      #progress_str = f'query mrr: {mean_mrr_q:.4f}, title mrr: {mean_mrr_t:.4f}'
      pbar.set_postfix({'query_mrr': f'{mean_mrr_q:.4f}', 
                       'title_mrr': f'{mean_mrr_t:.4f}' })
      #print('mrr query', f'{mean_mrr_q:.8f}', 
      #      'mrr title', f'{mean_mrr_t:.8f}')
    return t_ranks, q_ranks



#evaluator = Evaluator(name_uses, q_embeddings_to_use, title_embeddings)
#t_ranks, q_ranks = evaluator.eval(name_uses_test, model, 
#                                        filter_outputs=False, 
#                                        eval_sample_size=10_000)


#evaluator = Evaluator(name_uses, q_embeddings_to_use, title_embeddings)
#kk = list(title_embeddings.keys())
#cand = evaluator._get_candidates(kk[10], queries=True)
#self = evaluator


