"""A fixed crisis message for the voice loop, independent of the model.

A 4B model will sometimes get a crisis wrong, so check what the person said
(the speech-to-text transcript) before the model answers. If it matches, speak
the crisis message instead of the model's reply. ultron_voice.py does this.
False alarms ("I want to die of embarrassment") only cost one kind sentence.

Put your country's crisis line in CRISIS_LINE, or pass it to crisis_message()
(ultron_voice.py has --crisis-line). Write numbers the way they should be
spoken, for example "988" in the US or "Tele MANAS on one four four one six" in India.

    from safety_net import crisis_message
    message = crisis_message(transcript)
    if message:
        speak(message)
"""
import re

CRISIS_LINE = "a crisis line"


def message_for(crisis_line):
    return ("I'm stepping out of character, because this matters. "
            f"Please call {crisis_line} or your local emergency number right now, or tell someone near you. "
            "You don't have to go through this alone.")


CRISIS_MESSAGE = message_for(CRISIS_LINE)

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
    r"|goodbye (letter|note)|suicide note|last night alive|goodbye\b.*\b(for the last time|forever)"
    r"|waste of (space|air|oxygen)|(want|need) (it|this|everything) to stop"
    r")",
    re.IGNORECASE,
)


def crisis_message(transcript, crisis_line=None):
    """Return the crisis message if the transcript sounds like a risk of suicide or self-harm, else None."""
    if not _PATTERNS.search(transcript or ""):
        return None
    return message_for(crisis_line) if crisis_line else CRISIS_MESSAGE


if __name__ == "__main__":
    import sys
    for line in sys.stdin:
        print("MATCH  " if crisis_message(line) else "-      ", line.rstrip())
