"""Mask text before sending it to an LLM or another service, then put the original values back into the answer.

Each detected value gets a numbered placeholder ([EMAIL_1], [PERSON_2], ...); the same value always gets the same
placeholder, so the remote side can still refer to it. Only the placeholders leave your machine.
"""

from smith_pincette import SmithPincette


def pseudonymize(model, text):
    mapping, out, last, counts = {}, [], 0, {}
    by_value = {}
    for s in model.detect(text):
        key = (s.label, s.text)
        if key not in by_value:
            counts[s.label] = counts.get(s.label, 0) + 1
            by_value[key] = f"[{s.label.upper()}_{counts[s.label]}]"
            mapping[by_value[key]] = s.text
        out += [text[last:s.start], by_value[key]]
        last = s.end
    return "".join(out) + text[last:], mapping


def restore(text, mapping):
    for placeholder, value in mapping.items():
        text = text.replace(placeholder, value)
    return text


model = SmithPincette.from_pretrained("Holocortex/Smith-Pincette-Nitro", revision="v0.5-preview")
prompt = "Write a short reply to Jonas Weber (jonas.weber@example.com) confirming his order 88213 ships to Berlin."
safe_prompt, mapping = pseudonymize(model, prompt)
print("sent to the LLM:", safe_prompt)

llm_answer = safe_prompt.replace("Write a short reply to", "Dear")  # stand-in for a real LLM call
print("restored answer:", restore(llm_answer, mapping))
