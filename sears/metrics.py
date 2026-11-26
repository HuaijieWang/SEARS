import re
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .data import question_id, read_json


NUMBERS = {word: str(i) for i, word in enumerate("zero one two three four five six seven eight nine ten".split())}
NUMBERS["none"] = "0"
CONTRACTIONS = {
    word.replace("'", ""): word
    for word in (
        "ain't aren't can't could've couldn't didn't doesn't don't hadn't hasn't haven't he'd he's how'd how'll how's "
        "isn't it'd it'll ma'am mightn't might've mustn't must've needn't not've o'clock oughtn't shan't should've "
        "shouldn't somebody'll somebody's someone'd someone'll someone's something'd something'll that's there'd "
        "there're there's they'd they'll they're they've 'twas wasn't we've weren't what'll what're what's what've "
        "when's where'd where's where've who'd who'll who's who've why'll why're why's won't would've wouldn't y'all you'd "
        "you'll you're you've"
    ).split()
}
for word in (
    "couldn't've hadn't've he'd've it'd've mightn't've she'd've shouldn't've somebody'd've someone'd've "
    "something'd've there'd've they'd've we'd've who'd've wouldn't've y'all'll y'all'd've you'd've"
).split():
    for index, char in enumerate(word):
        if char == "'":
            CONTRACTIONS[word[:index] + word[index + 1:]] = word
CONTRACTIONS.update({"somebody'd": "somebodyd", "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at"})


def vqa_normalize(text):
    text = text.lower().replace(",", "").replace("?", "").replace("'s", " 's").replace("\n", " ").replace("\t", " ").strip()
    original = text
    for char in ';/[]"{}()=+\\_-><@` ,?!'.replace(" ", ""):
        if char + " " in original or " " + char in original:
            text = text.replace(char, "")
        else:
            text = text.replace(char, " ")
    text = re.sub(r"\.(?!\d)", "", text)
    words = (NUMBERS.get(word, word) for word in text.split())
    return " ".join(CONTRACTIONS.get(word, word) for word in words if word not in {"a", "an", "the"})


def textvqa_score(prediction, answers):
    target = vqa_normalize(prediction)
    answers = [vqa_normalize(answer) for answer in answers]
    matches = answers.count(target)
    return mean(min(1.0, (matches - int(answer == target)) / 3.0) for answer in answers)


def levenshtein(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def anls(prediction, references):
    prediction = prediction.strip().lower()
    def similarity(reference):
        reference = reference.strip().lower()
        size = max(len(prediction), len(reference))
        return 1.0 - levenshtein(prediction, reference) / size if size else 1.0
    return max(similarity(value) for value in (references if isinstance(references, list) else [references]))


def rec_score(text, target):
    try:
        text = text.replace("[", "").replace("]", "")
        x1, y1, x2, y2 = (float(value) for value in text[1:-1].split(","))
        if isinstance(target, str):
            target = [float(value) for value in target.strip("[]").split(",")]
        a1, b1, a2, b2 = target
    except (TypeError, ValueError):
        return 0.0
    intersection = max(0, min(x2, a2) - max(x1, a1)) * max(0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return float(union > 0 and intersection / union > 0.5)


def scienceqa_answer(text, choices):
    options = "ABCDE"[:len(choices)]
    if len(text) == 1 and text in options:
        return options.index(text)
    if len(text) >= 3 and text[0] in options and text[1:3] == ". ":
        return options.index(text[0])
    matches = re.findall(r"The answer is ([A-Z]).", text)
    return options.index(matches[0]) if len(matches) == 1 and matches[0] in options else -1


def ptb_tokenize(captions_by_image):
    from pycocoevalcap.tokenizer import ptbtokenizer as ptb

    jar = Path(ptb.__file__).resolve().with_name(ptb.STANFORD_CORENLP_3_4_1_JAR)
    pairs = [(key, entry["caption"].replace("\n", " ")) for key, entries in captions_by_image.items() for entry in entries]
    command = ["java", "-cp", str(jar), "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase"]
    payload = "\n".join(text for _, text in pairs).encode("utf-8")
    output = subprocess.run(command, input=payload, stdout=subprocess.PIPE, check=True).stdout
    tokenized = defaultdict(list)
    punctuation = set(ptb.PUNCTUATIONS)
    for (key, _), line in zip(pairs, output.decode("utf-8").split("\n")):
        tokenized[key].append(" ".join(word for word in line.rstrip().split(" ") if word not in punctuation))
    return dict(tokenized)


def captions(rows, predictions, metric, annotations):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge

    if metric == "cider":
        references = {question_id(row, i): row["answer"] for i, row in enumerate(rows)}
        outputs = {question_id(row, i): [text] for i, (row, text) in enumerate(zip(rows, predictions))}
        selected = {}
        scorer = Cider()
        for id, output in outputs.items():
            refs = references[id]
            candidate_refs = {i: [ref] for i, ref in enumerate(refs)}
            _, scores = scorer.compute_score(candidate_refs, dict.fromkeys(candidate_refs, output))
            selected[id] = candidate_refs[max(candidate_refs, key=lambda index: scores[index])]
        return 100 * float(scorer.compute_score(outputs, selected)[0])
    truth = defaultdict(list)
    for row in annotations["annotations"]:
        truth[int(row["image_id"])].append({"caption": row["caption"]})
    predictions = {i: [{"caption": text}] for i, text in enumerate(predictions, 1)}
    truth, predictions = ptb_tokenize(truth), ptb_tokenize(predictions)
    cider = Cider().compute_score(truth, predictions)[0]
    bleu = Bleu(4).compute_score(truth, predictions)[0]
    meteor = Meteor().compute_score(truth, predictions)[0]
    rouge = Rouge().compute_score(truth, predictions)[0]
    return 100 * mean([*bleu, meteor, rouge, cider])


def score(metric, rows, predictions, annotation_path=None):
    annotations = read_json(annotation_path) if annotation_path else None
    if metric in {"caption", "cider"}:
        return captions(rows, predictions, metric, annotations)
    ids = [question_id(row, i) for i, row in enumerate(rows)]
    by_id = dict(zip(ids, rows))
    if metric == "scienceqa":
        subset = read_json(annotation_path.with_name("pid_splits.json"))["test"]
        outputs = dict(zip(ids, predictions))
        correct = sum(scienceqa_answer(outputs[id], annotations[id]["choices"]) == annotations[id]["answer"] for id in subset if id in outputs)
        return 100 * correct / len(outputs)
    if metric == "textvqa":
        target = {str(row["question_id"]): row["answers"] for row in annotations["data"]}
        return 100 * mean(textvqa_score(text, target[id]) for id, text in zip(ids, predictions))
    if metric == "gqa":
        outputs = dict(zip(ids, predictions))
        return 100 * mean(text.strip().lower().rstrip(".") == annotations[id]["answer"] for id, text in outputs.items() if annotations[id]["isBalanced"])
    values = []
    for index, (id, text) in enumerate(zip(ids, predictions)):
        if metric == "imagenet":
            label = rows[index]["answer"]
            value = label in text or text in label
        elif metric == "anls":
            value = anls(text, by_id[id]["answer"])
        elif metric == "rec":
            value = rec_score(text, by_id[id]["answer_bbox"])
        elif metric == "ocrvqa":
            value = "Unanswerable" not in text and text.lower() == by_id[id]["answer"].lower()
        else:
            value = text.upper() == by_id[id]["answer"].upper()
        values.append(value)
    return 100 * mean(values)


def aggregate(matrix, run):
    final = matrix[-1]
    diagonal = [row[i] for i, row in enumerate(matrix)]
    results = {
        "MFT": mean(diagonal),
        "MFN": mean(final),
        "MAA": mean(mean(row) for row in matrix),
        "BWT": mean(a - b for a, b in zip(final[:-1], diagonal[:-1])),
    }
    if run == "cvit-llava":
        results["AP"], results["MAP"] = results["MFN"], results["MAA"]
    return results
