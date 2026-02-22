# 🧠 Reflection with External Feedback: Why LLMs Get Smarter When They Can Check Themselves

Large Language Models (LLMs) are powerful, but they are not reliable reasoners by default.  
They often produce answers that *sound confident but are wrong*, especially for tasks involving logic, constraints, or edge cases.

In this post, I’ll show a **small, reproducible experiment** that demonstrates why:

> 👉 *Reflection combined with external feedback dramatically improves LLM accuracy compared to pure prompting.*

We’ll compare:
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

We first query the LLM directly and trust its output.

```python
PROMPT_TEMPLATE = """
{prefix}
Given a list of integers and a number k, return the kth largest element in the array.
If it's not possible to find the kth largest element, return -1.
Array: {nums}
K: {k}"""
```

### ❌ Observed Behaviour

- The model is often correct  
- But it fails on edge cases and invalid `k` values  

**Observed accuracy: ~50%**

---

## 🔁 Adding Reflection with External Feedback

We introduce **reflection with a verifier**.

```python
def check_kth_largest_element(nums_array: list[int], k: int, llm_response: KthLargestElement) -> bool:
    sorted_nums = sorted(nums_array, reverse=True)
    k_th_largest_element = sorted_nums[k - 1] if 1 <= k <= len(sorted_nums) else -1
    return k_th_largest_element == llm_response.kth_largest_element
```

This function acts as:
> ✅ A ground-truth oracle

---

## 🪞 Reflection Prompt

```python
def make_prefix_prompt(previous_output: KthLargestElement) -> str:
    return f"""
    Your previous output was incorrect. Try again.
    Previous output: {previous_output.kth_largest_element}
    """
```

This turns the LLM into a **self-correcting loop**.

---

## 📊 Results

| Setup              | Accuracy |
|--------------------|----------|
| Without reflection | 50%      |
| With reflection    | 100%     |

---

## 🔑 Key Takeaway

> Reflection without feedback is weak.  
> Reflection with **external verification** is powerful.

This is the difference between prompting and **engineering reliable LLM systems**.
