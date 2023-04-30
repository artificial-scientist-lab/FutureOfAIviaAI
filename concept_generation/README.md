Here we demonstrate how our concept list is generated.

### 1) Initial Concept list
The file RAKE_create_concept_list_from_papers.py generates from a list of arxiv papers (arxiv_data_new.pkl) an initial list (full_concepts_20210320.pkl).

### 2) Refining the concept list
The file reduce_concept_list.py contains numerous hand-crafted criteria when concept proposals are not actual concepts. This leads to full_concepts_new.txt.
