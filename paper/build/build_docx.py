"""Assemble the manuscript from section files, number citations, convert to .docx with pandoc and post-process (tables, line numbers, page numbers)."""
import re, os, subprocess, sys, copy
import pypandoc
sys.path.insert(0, os.path.dirname(__file__)); from refs import REFS
B = os.path.dirname(os.path.abspath(__file__)); os.chdir(B)

def split_h2(text):
    parts = re.split(r"(?m)^(?=## )", text); head = parts[0]; d = {}
    for p in parts[1:]: d[p.split("\n", 1)[0][3:].strip()] = p
    return head, d
def read(f): return open(f).read()

a_head, a = split_h2(read("sec06a_results.md"))
res_order = ["Random versus group-held-out evaluation", "Learning curves and extrapolation"]
parts = [read("sec00_front.md"), read("sec01_intro.md"), read("sec02_related.md"), read("sec03_theory.md"), read("sec04_data.md"), read("sec05_design.md"), a_head]
parts += [a[k] for k in res_order]
for f in ("sec06b_parkinsons.md", ): 
    if os.path.exists(f): parts.append(read(f))
parts.append(a["Concrete: leakage is concentrated in rows with group-mates"])
if os.path.exists("sec06d_theory_data.md"): parts.append(read("sec06d_theory_data.md"))
parts.append(read("sec06c_tuning.md"))
if os.path.exists("sec06e_sensitivity.md"): parts.append(read("sec06e_sensitivity.md"))
parts.append(a["Ranking reversal across protocols"])
parts += [read("sec07_discussion.md"), read("sec08_conclusion.md"), read("sec09_appendix.md")]
text = "\n\n".join(p.rstrip() for p in parts) + "\n"
# unnumbered headings
for h in ("Declarations", "Appendix A. Derivations", "Appendix B. Reproducibility details"): text = text.replace(f"# {h}\n", f"# {h} {{-}}\n")
# citations -> numbers by first appearance
order = []
def cite(m):
    keys = re.findall(r"@([A-Za-z0-9_]+)", m.group(0)); nums = []
    for k in keys:
        if k not in REFS: raise KeyError(k)
        if k not in order: order.append(k)
        nums.append(order.index(k) + 1)
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1: j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1])); i = j + 1
    return "[" + ", ".join(out) + "]"
text = re.sub(r"\[@[^\]]+\]", cite, text)
refs = "\n\n".join(f"[{i+1}] {REFS[k]}" for i, k in enumerate(order))
text += "\n\n# References {-}\n\n" + refs.replace("[", "\\[").replace("]", "\\]") + "\n"
open("manuscript.md", "w").write(text)
words = len(re.findall(r"\w+", re.sub(r"!\[.*?\]\(.*?\)", "", text.split("# References")[0])))
print("citations:", len(order), "| words in markdown body (incl. tables, equations text):", words)

extra = ["--from=markdown+tex_math_dollars+pipe_tables+fenced_divs+implicit_figures", "--number-sections", "--resource-path=.:..", "--reference-doc=reference.docx", "--wrap=none"]
pypandoc.convert_file("manuscript.md", "docx", outputfile="manuscript_raw.docx", extra_args=extra)
print("pandoc ok")
