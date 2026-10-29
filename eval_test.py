from datasets import load_dataset
from retrieval import hybrid_search, search_qdrant
from config import CHAT_MODEL, FIXED_COLLECTION, COLLECTION_NAME
from embeddings import client
from generate import answer
import re

def recall_at_k(gold_titles, retrieved_titles):
        if not gold_titles:
            return None
        return len(gold_titles & set(retrieved_titles)) / len(gold_titles)


def reciprocal_rank(gold_titles, retrieved_titles):
    for pos, title in enumerate(retrieved_titles, start=1):
       if title in gold_titles: 
            return 1 / pos
    return 0.0


def token_f1(predicted, gold):
    pred_tokens = set(predicted.lower().split())
    gold_tokens = set(gold.lower().split())
    common = pred_tokens & gold_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def llm_judge(question, predicted, gold):
    prompt = (
        f"Question: {question}\n"
        f"Correct answer: {gold}\n"
        f"Predicted answer: {predicted}\n\n"
        "Is the predicted answer correct? Consider meaning, not exact wording. "
        "Answer only 'correct' or 'incorrect'."
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
    )
    verdict = resp.choices[0].message.content.strip().lower()
    return 1.0 if "correct" in verdict else 0.0



def refusal_judge(answer):
    prompt = (
        f"Predicted answer: {answer}\n\n"
        "Does this response refuse to answer or state that the information is not available? Answer strictly only 'yes' or 'no'"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
    )
    verdict = resp.choices[0].message.content.strip().lower()
    return 1.0 if "yes" in verdict else 0.0

def contra_judge(answer):
    prompt = (
        f"Predicted answer: {answer}\n\n"
        "Does this response flag a contradiction or conflict between different sources? Answer strictly only 'yes' or 'no'.'"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="none",
    )
    verdict = resp.choices[0].message.content.strip().lower()
    return 1.0 if "yes" in verdict else 0.0




def citation_accuracy(answer_text, all_hits, gold_titles):
    cited_nums = [int(n) for n in re.findall(r"\[(\d+)\]", answer_text)]
    if not cited_nums:
        return 0.0
    correct = 0
    for n in cited_nums:
        if n <= len(all_hits):
            hit = all_hits[n - 1]
            title = hit.payload.get("title", hit.payload.get("name", ""))
            if title in gold_titles:
                correct += 1
    return correct / len(cited_nums)


# eval set

NO_ANSWER_QUESTIONS = [
    {"question": "What is the population of Mars colony in 2025?", "answer": None, "gold_titles": set(), "type": "no-answer"},
    {"question": "Who won the 2030 FIFA World Cup?", "answer": None, "gold_titles": set(), "type": "no-answer"},
    {"question": "What is the latest Anthropic Claude model?", "answer": None, "gold_titles": set(), "type": "no-answer"},
    {"question": "What is the speed of dark matter particles?", "answer": None, "gold_titles": set(), "type": "no-answer"},
    {"question": "Which OpenAI GPT models have been deprecated?", "answer": None, "gold_titles": set(), "type": "no-answer"},
    {"question": "What is the chemical formula of unobtanium?", "answer": None, "gold_titles": set(), "type": "no-answer"},
]

CONTRADICTORY_QUESTIONS = [
    {"question": "What nationality was Scott Derrickson?", "answer": "contradictory", "gold_titles": {"Scott Derrickson"}, "type": "contradictory"},
    {"question": "What nationality was Ed Wood?", "answer": "contradictory", "gold_titles": {"Ed Wood"}, "type": "contradictory"},
    {"question": "Did Shirley Temple hold any government positions?", "answer": "contradictory", "gold_titles": {"Shirley Temple"}, "type": "contradictory"},
    {"question": "Who portrayed Corliss Archer in the film Kiss and Tell?", "answer": "contradictory", "gold_titles": {"Kiss and Tell (1945 film)"}, "type": "contradictory"},
    {"question": "During what years did Alex Ferguson manage Manchester United?", "answer": "contradictory", "gold_titles": {"Alex Ferguson"}, "type": "contradictory"},
    {"question": "Where is the main campus of the University of Kansas located?", "answer": "contradictory", "gold_titles": {"University of Kansas"}, "type": "contradictory"},
]


def build_eval_set(n=28):

    data = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True).take(n)
    eval_set =[]
    for item in data:
        eval_set.append({
            "question": item["question"],
            "answer": item["answer"],
            "gold_titles": set(item["supporting_facts"]["title"]),
            "type": "multi-hop",
        })
    eval_set.extend(NO_ANSWER_QUESTIONS)
    eval_set.extend(CONTRADICTORY_QUESTIONS)
    return eval_set


def run_eval(retriever="hybrid", k=5, collection_name=COLLECTION_NAME):

    eval_set = build_eval_set()
    results = []

    for i, item in enumerate(eval_set):
        print(f"[{i+1}/{len(eval_set)}] {item['question']}")
         # retrieval

        if retriever == "hybrid":
            hits = hybrid_search(item["question"], k=k, collection_name=collection_name)
        else:
            hits = search_qdrant(item["question"], k=k, collection_name=collection_name)

        if hits is None:
            hits = []

        retrieved_titles = [h.payload.get("title", h.payload.get("name", "")) for h in hits]

        # retrieval metrics
        rec = recall_at_k(item["gold_titles"], retrieved_titles)
        rr = reciprocal_rank(item["gold_titles"], retrieved_titles)

        # generation
        answer_text, all_hits = answer(item["question"], collection_name=collection_name)

         # answer correctness
        if item["answer"] is None:
            # no-answer: correct if system refuses
            refused = refusal_judge(answer_text)
            correctness = 1.0 if refused else 0.0
            f1 = None
            judge = None
        elif item["answer"] == "contradictory":
            # contradictory: correct if system flags conflict
            flagged = contra_judge(answer_text)
            correctness = 1.0 if flagged else 0.0
            f1 = None
            judge = None
        else:
            f1 = token_f1(answer_text, item["answer"])
            judge = llm_judge(item["question"], answer_text, item["answer"])
            correctness = judge

        # citation accuracy
        cit_acc = citation_accuracy(answer_text, all_hits, item["gold_titles"])

        result = {
            "question": item["question"],
            "type": item["type"],
            "recall": rec,
            "mrr": rr,
            "token_f1": f1,
            "llm_judge": judge,
            "correctness": correctness,
            "citation_accuracy": cit_acc,
        }

        results.append(result)
        print(f"  recall={rec} mrr={rr:.2f} correct={correctness} cit_acc={cit_acc:.2f}")
        print()
 # --- Summary ---
    print("=" * 60)
    print(f"RETRIEVER: {retriever}")
    print(f"Total questions: {len(results)}")

    for qtype in ["multi-hop", "no-answer", "contradictory"]:
        subset = [r for r in results if r["type"] == qtype]
        if not subset:
            continue
        print(f"\n--- {qtype} ({len(subset)} questions) ---")
        recalls = [r["recall"] for r in subset if r["recall"] is not None]
        mrrs = [r["mrr"] for r in subset if r["mrr"] is not None]
        corrs = [r["correctness"] for r in subset]
        cits = [r["citation_accuracy"] for r in subset]
        if recalls:
            print(f"  Avg Recall@{k}: {sum(recalls)/len(recalls):.3f}")
        if mrrs:
            print(f"  Avg MRR:        {sum(mrrs)/len(mrrs):.3f}")
        print(f"  Avg Correctness: {sum(corrs)/len(corrs):.3f}")
        print(f"  Avg Citation Acc: {sum(cits)/len(cits):.3f}")

    # overall
    all_corr = [r["correctness"] for r in results]
    all_cit = [r["citation_accuracy"] for r in results]
    all_rec = [r["recall"] for r in results if r["recall"] is not None]
    all_mrr = [r["mrr"] for r in results if r["mrr"] is not None]
    print(f"\n--- OVERALL ---")
    print(f"  Avg Recall@{k}: {sum(all_rec)/len(all_rec):.3f}")
    print(f"  Avg MRR:        {sum(all_mrr)/len(all_mrr):.3f}")
    print(f"  Avg Correctness: {sum(all_corr)/len(all_corr):.3f}")
    print(f"  Avg Citation Acc: {sum(all_cit)/len(all_cit):.3f}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "fixed":
        from config import FIXED_COLLECTION
        print("=== FIXED CHUNKING (hybrid) ===")
        run_eval(retriever="hybrid", collection_name=FIXED_COLLECTION)
        print("\n\n=== DENSE-ONLY ===")
        run_eval(retriever="dense", collection_name=FIXED_COLLECTION)
    else:
        print("=== PARAGRAPH CHUNKING (hybrid) ===")
        run_eval(retriever="hybrid")
        print("\n\n=== DENSE-ONLY ===")
        run_eval(retriever="dense")