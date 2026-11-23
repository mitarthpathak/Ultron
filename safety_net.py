"""A fixed crisis message for the voice loop, independent of the model.

A 4B model will sometimes get a crisis wrong, so check what the person said
(the speech-to-text transcript) before the model answers. If it matches, speak
CRISIS_MESSAGE first, then the model's reply. False alarms ("I want to die of
embarrassment") only cost one kind sentence.

Put your country's crisis line in CRISIS_LINE (for example 988 in the US).

    from safety_net import crisis_message
    extra = crisis_message(transcript)
    if extra:
        speak(extra)
"""
import re

CRISIS_LINE = "a crisis line"

CRISIS_MESSAGE = (
    "I'm stepping out of character, because this matters. "
    f"Please call {CRISIS_LINE} or your local emergency number right now, or tell someone near you. "
    "You don't have to go through this alone."
)

_PATTERNS = re.compile(
    r"\b("
    r"kill(ing)? my ?self|suicid\w*|end(ing)? (my life|it all|my ?self)"
    r"|(want|wanna|going|ready|deserve) to die|should (just )?die|wish (i was|i were|i'?d) (dead|never born)"
    r"|want (it|this|my life|everything) to (end|be over)|point (of|in) (living|life|being alive)"
    r"|if i (died|disappeared|was gone|were gone|wasn'?t here)|thinking about jumping"
    r"|(pills|tablets|meds)\b.*\b(all|every)\b|\b(all|every)\b.*\b(pills|tablets|meds)\b"
    r"|(do ?n[o']?t|no longer) want to (live|be alive|exist|be here|wake up)"
    r"|tired of (living|being alive)|stop existing|better off (without me|if i (was|were) (dead|gone))"
    r"|hurt(ing)? my ?self|harm(ing)? my ?self|self[- ]?harm|cut(ting)? my ?self"
    r"|overdos\w*|took (a lot of|too many|all (of )?(my|the)) (pills|meds|medicine|medication|tablets)"
    r"|goodbye (letter|note)|suicide note|last night alive"
    r")",
    re.IGNORECASE,
)


def crisis_message(transcript):
    """Return CRISIS_MESSAGE if the transcript sounds like a risk of suicide or self-harm, else None."""
    return CRISIS_MESSAGE if _PATTERNS.search(transcript or "") else None


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        print("MATCH  " if crisis_message(line) else "-      ", line.rstrip())
