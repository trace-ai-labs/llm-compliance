"""Word-level variants for the wording ablation."""

# Obligation verb (or phrase) used in the rule sentence, weakest to strongest spread.
OBLIGATION_VARIANTS = {
    "requires": "requires",
    "mandates": "mandates",
    "must_use": "must use",
    "should_use": "should use",
    "expects": "expects",
    "recommends": "recommends",
    "encourages": "encourages",
}

# Noun used for the monetary penalty, from legal-sanction to market-transaction framing.
PENALTY_WORDS = ["fine", "penalty", "fee", "charge", "surcharge"]
