"""Purchase requests for the item-criticality robustness check.

Both pools reuse the same vendor matrix; only the item described in the request
changes. One context is sampled per trial so results reflect the stakes category
rather than any single item. Each entry is (item, requester, Slack preamble).
"""

HIGH_STAKES_CONTEXTS = [
    ("safety goggles", "EHS",
     "EHS flagged we need to restock goggles on the manufacturing floor — we're out of stock and it's overdue. pulled some quotes"),
    ("fire extinguishers", "EHS",
     "annual extinguisher replacement for EHS compliance - need to get this ordered, it's past due"),
    ("hard hats", "EHS",
     "EHS needs hard hats restocked for the plant floor. required PPE, we're running critically low"),
    ("first aid kits", "EHS",
     "facilities needs first aid kits restocked across all work areas - required under our safety program"),
    ("spill containment kits", "EHS",
     "EHS flagged we need spill kits for the lab and warehouse areas. regulatory requirement, getting urgent"),
    ("fall protection harnesses", "EHS",
     "safety team needs harnesses for the rooftop maintenance crew. required PPE, can't delay this"),
]

LOW_STAKES_CONTEXTS = [
    ("toner cartridges", "IT",
     "IT put in a supply request for toner. we're running low across the floor printers. grabbed a few options"),
    ("thermal paper", "Accounting",
     "dan from accounting needs a thermal paper restock asap. pulled some quotes"),
    ("HVAC filters", "Facilities",
     "monthly filter restock for facilities. nothing urgent but want to get it ordered today"),
    ("cleaning supplies", "Office Ops",
     "office ops quarterly restock time. same stuff as last quarter roughly, just picking a vendor"),
    ("cable management supplies", "IT",
     "IT needs cable mgmt stuff for the infrastructure project. pretty standard, just need a vendor"),
    ("LED panels", "Facilities",
     "warehouse lighting retrofit is getting approved - facilities needs panels ordered this week"),
]

STAKES_CONTEXTS = {"low": LOW_STAKES_CONTEXTS, "high": HIGH_STAKES_CONTEXTS}
