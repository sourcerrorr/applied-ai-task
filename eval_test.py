from datasets import load_dataset
from retrieval import search_qdrant

def recall_at_k(gold_titles, retrieved_titles):
        return len(gold_titles & set(retrieved_titles)) / len(gold_titles)


def reciprocal_rank(gold_titles, retrieved_titles):
    for pos, title in enumerate(retrieved_titles, start=1):
       if title in gold_titles: 
            return 1 / pos
    return 0.0
        

def build_eval_set(n=8):

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True).take(n)
    gold_list =[]
    for item in data:
        gold_dict = {}
        gold_dict["question"] = item["question"]
        gold_dict["gold_titles"] =set(item["supporting_facts"]["title"])
        gold_list.append(gold_dict)
    return gold_list

def run_eval(k=5):

    recalls = []
    rrs = []
    eval_set = build_eval_set()
    for item in eval_set:
        hits = search_qdrant(item["question"], k)
        retrieved_titles = [hit.payload["title"] for hit in hits]
        recalls.append(recall_at_k(item["gold_titles"], retrieved_titles))
        rrs.append(reciprocal_rank(item["gold_titles"], retrieved_titles))
        print("Q:", item["question"])
        print("  gold:     ", item["gold_titles"])
        print("  retrieved:", retrieved_titles)
        print()

    print("Avg recall: ", sum(recalls)/len(recalls))
    print("MRR: ", sum(rrs)/len(rrs) )