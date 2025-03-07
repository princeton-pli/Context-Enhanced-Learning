import numpy as np
import json


class TEMPLATE:

    str_seperator = " "

    task_name = "Xasijeness"

    system_prompt = "You are a helpful language translator."

    eot_token = "<|eot_id|>"

    language_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']    

    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'α', 'β', 'Γ', 'γ', 'Δ', 'δ', 'ε', 'ζ', 'η', 'θ', 'ι', 'κ', 'Λ', 'λ', 'μ', 'ν', 'Ξ', 'ξ', 'Π', 'π', 'Σ', 'σ', 'τ', 'υ', 'Φ', 'φ', 'χ', 'Ψ', 'ψ', 'Ω', 'ω', 'क', 'ख', 'ग', 'घ', 'च', 'छ', 'ज', 'झ', 'ट', 'ठ', 'ड', 'ढ', 'त', 'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'स', 'ह', 'б', 'в', 'г', 'д', 'ж', 'з', 'к', 'л', 'м', 'н', 'п', 'т', 'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'и', 'э', 'ю', 'я']

    abs_letters = [f'<|reserved_special_token_{i}|>' for i in range(110, 230)]

    task_header = "You are performing a special translation task called {task_name}."

    rule_description = "The task translates a token sequence in language {lang_source} through multiple abstract languages using a chain of {d} dictionaries. For each round of translation, you should first trims the first and last token and then iterates through the corresponding dictionary, finding and translating the longest matching subsequences. This trimming + translation process repeats for each language pair, ultimately producing a sequence in the language {lang_dest}."

    rule_description_claude = "The task translates a token sequence through multiple abstract languages using a chain of {d} dictionaries from language {lang_source} to language {lang_dest}. Starting with an unindented sequence in the initial language, it first trims the first and last token. It then iterates through each dictionary, finding and translating the longest matching subsequences. This process repeats for each language pair, ultimately producing a sequence in language {lang_dest}."

    rule_description_gpt4o = "The task involves translating a token sequence from language {lang_source} to language {lang_dest} using {d} dictionaries. At each step, the first and last tokens are removed, the longest matching sequence is found in the current dictionary, and it’s translated to the next language. This repeats through all {d} dictionaries until the sequence reaches language {lang_dest}."

    dictionaries_header = "The set of dictionaries used are as follows:"

    dictionaries_header_subset = "The subset of dictionaries used are as follows:"

    dictionary_desc = "\nDictionary used from language {lang_in} to language {lang_out}:\n{dictionary}"

    mapping_template = "{tuple_source} ->{tuple_dest};"

    q_template = "Please perform {task_name} translation from the following sequence in language {lang_source} to language {lang_dest}. The sequence you need to translate is:"

    q_template = """Now please perform {task_name} translation from the following sequence in language {lang_source} to language {lang_dest}. Do not use code! You must only reponse in the form: "{cot_template_example}". The sequence you need to translate from language {lang_source} is:{q_seq}."""

    cot_template = "Sequence in language {lang}:{lang_str};"

    cot_demo_template = " [the sequence in language {lang}]"

    a_template = "The translated result in language {lang_dest} is:"

    a_cot = "Language {lang_tmp}: {padded_tmp_result};"

    a_template_cot = "The translation result is: {cot_answer}"

    a_cot_max_length = 100

    think_token = "<|reserved_special_token_100|>"
