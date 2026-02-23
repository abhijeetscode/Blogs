# 🧠 Reflection with External Feedback: Why LLMs Get Smarter When They Can Check Themselves

Large Language Models (LLMs) are powerful, but they are not reliable reasoners by default.  
They often produce answers that *sound confident but are wrong*, especially for tasks involving logic, constraints, or edge cases.

In this post, I'll show a **small, reproducible experiment** that demonstrates why:

> 👉 *Reflection combined with external feedback dramatically improves LLM accuracy compared to pure prompting.*

We compare:
- ❌ LLM without reflection (pure prompting)
- ✅ LLM with reflection + external verification

---

## 🎯 Problem Setup: Kth Largest Element

We ask the LLM:

> Given a list of integers and a number `k`, return the kth largest element in the array.  
> If it's not possible to find the kth largest element, return `-1`.

This is intentionally simple, yet it exposes typical LLM failure modes:
- Off-by-one errors  
- Incorrect sorting  
- Failure on edge cases (`k > len(array)`)  

---

## 🧪 Baseline: LLM Without Reflection (Pure Prompting)

```python
PROMPT_TEMPLATE = """
{prefix}
Given a list of integers and a number k, return the kth largest element in the array.
If it's not possible to find the kth largest element, return -1.
Array: {nums}
K: {k}"""
```

**Observed behaviour:**
- Sometimes correct
- Often fails on edge cases

Accuracy observed: **~50%**

---

## 🔁 Adding Reflection with External Feedback

We introduce a deterministic verifier:

```python
def check_kth_largest_element(nums_array: list[int], k: int, llm_response: KthLargestElement) -> bool:
    sorted_nums = sorted(nums_array, reverse=True)
    k_th_largest_element = sorted_nums[k - 1] if 1 <= k <= len(sorted_nums) else -1
    return k_th_largest_element == llm_response.kth_largest_element
```

This function acts as a **ground-truth oracle**.  
The model is no longer trusted blindly.

---

## 🪞 Reflection Prompt

```python
def make_prefix_prompt(previous_output: KthLargestElement) -> str:
    return f"""
    Your previous output was incorrect. Try again.
    Previous output: {previous_output.kth_largest_element}
    """
```

This enables a simple loop:

1. Generate answer  
2. Verify externally  
3. Reflect failure  
4. Retry  

> 🔁 A maximum of **3 retries** are allowed per test case. If the model fails all attempts, the last output is taken as the final answer.

This is the core of **agentic behaviour**.

---

## 📊 Results

Below is the accuracy comparison from the experiment:

![Accuracy comparison: without reflection vs with reflection](accuracy_comparison.png)

| Setup              | Accuracy | Max Retries |
|--------------------|----------|-------------|
| Without reflection | 50%      | —           |
| With reflection    | 100%     | 3           |

Reflection + external feedback **doubles accuracy** by closing the loop between the model and the environment.

---

## 🧠 Why This Works

Pure prompting:
- No notion of correctness  
- No learning from failure  

Reflection + external feedback:
- Adds a verifier (critic)  
- Enables iterative improvement  
- Produces more reliable outputs  

This mirrors how humans solve problems:
> Try → Check → Reflect → Fix

---

## 🤖 This Pattern Powers Agentic Systems

This tiny example is the foundation of:
- ReAct agents  
- Tool-using LLMs  
- Self-correcting pipelines  
- Critic/Verifier architectures  

---

## 🧩 Full Reproducible Code

```python
import warnings
warnings.filterwarnings("ignore")
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from typing import cast
import matplotlib.pyplot as plt

class KthLargestElement(BaseModel):
    kth_largest_element: int = Field(description="The kth largest element in the array")

load_dotenv()

SEED = 42  # set for reproducible outputs
model = "llama3:8b"
llm = ChatOllama(
    model=model,
    seed=SEED,
).with_structured_output(KthLargestElement)

test_set: list[dict[str, int|list[int]]] = [
    {"nums": [3, 2, 1, 5, 6, 4], "k": 3, "expected": 4},
    {"nums": [3, 2, 1, 5, 6, 4], "k": 6, "expected": 1},
    {"nums": [3, 2, 1, 5, 6, 4], "k": 7, "expected": -1},
    {"nums": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "k": 1, "expected": 10},
    {"nums": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "k": 10, "expected": 1},
    {"nums": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "k": 11, "expected": -1},
]

PROMPT_TEMPLATE = """
{prefix}
Given a list of integers and a number k, return the kth largest element in the array.
If it's not possible to find the kth largest element, return -1.
Array: {nums}
K: {k}"""

print("==== Without reflection ====")
correct_count = 0
for i, case in enumerate(test_set):
    nums, k, expected = case["nums"], case["k"], case["expected"]
    prompt: str = PROMPT_TEMPLATE.format(prefix="", nums=nums, k=k)
    result = llm.invoke([
        HumanMessage(content=prompt),
    ])
    result = cast(KthLargestElement, result)
    ok = result.kth_largest_element == expected
    if ok:
        correct_count += 1
    status = "✅" if ok else "❌"
    print(f"{status} case {i + 1}: nums={nums}, k={k} -> got {result.kth_largest_element}, expected {expected}")

accuracy_without = 100 * correct_count / len(test_set)
print(f"\nAccuracy: {correct_count}/{len(test_set)} ({accuracy_without:.0f}%)")

print("==== With reflection ====")
def check_kth_largest_element(nums_array: list[int], k: int, llm_response: KthLargestElement) -> bool:
    sorted_nums = sorted(nums_array, reverse=True)
    k_th_largest_element = sorted_nums[k - 1] if 1 <= k <= len(sorted_nums) else -1
    return k_th_largest_element == llm_response.kth_largest_element

def make_prefix_prompt(previous_output: KthLargestElement) -> str:
    return f"""
    Your previous output was incorrect. Try again.
    Previous output: {previous_output.kth_largest_element}
    """

correct_count = 0
for i, case in enumerate(test_set):
    nums, k, expected = case["nums"], case["k"], case["expected"]
    prompt = PROMPT_TEMPLATE.format(prefix="", nums=nums, k=k)
    result = llm.invoke([
        HumanMessage(content=prompt),
    ])
    result = cast(KthLargestElement, result)
    ok: bool = check_kth_largest_element(nums_array=nums, k=k, llm_response=result)
    if not ok:
        attempts: int = 0
        MAX_ATTEMPTS: int = 3
        while not ok and attempts < MAX_ATTEMPTS:
            print(f"Attempt {attempts + 1} of {MAX_ATTEMPTS} failed. Retrying...")
            result = cast(KthLargestElement, llm.invoke([
                HumanMessage(content=PROMPT_TEMPLATE.format(prefix=make_prefix_prompt(result), nums=nums, k=k)),
            ]))
            ok = check_kth_largest_element(nums_array=nums, k=k, llm_response=result)
            attempts += 1
    if ok:
        correct_count += 1
    status = "✅" if ok else "❌"
    print(f"{status} case {i + 1}: nums={nums}, k={k} -> got {result.kth_largest_element}, expected {expected}")

accuracy_with = 100 * correct_count / len(test_set)
print(f"\nAccuracy: {correct_count}/{len(test_set)} ({accuracy_with:.0f}%)")

fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Without reflection", "With reflection"]
accuracies = [accuracy_without, accuracy_with]
bars = ax.bar(labels, accuracies, edgecolor="black", linewidth=1.2)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{acc:.0f}%", ha="center", fontweight="bold")
plt.tight_layout(rect=(0, 0, 1, 0.92))
plt.savefig("accuracy_comparison.png", dpi=150)
plt.show()
```
