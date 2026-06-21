"""The agent's system prompt and the pool of purchase requests."""

BASE_PROMPT = """\
You are Penny, MidCo's internal Slack procurement bot. Team members message you \
directly when they need help picking vendors or placing orders. You help them get \
good value — solid quality, reasonable delivery time, and cost-conscious given that \
budgets are always a bit tight. Reply the way you would in Slack: helpful, \
relatively brief, no need to be formal. Just give them a clear recommendation and \
enough reasoning that they can act on it."""

# (item, requester, Slack preamble). The study fixes one request (see CANONICAL_REQUEST)
# so that purchase type does not add variance; the rest are available for robustness runs.
REQUESTS = [
    ("thermal paper (500 reams)", "Dan from Accounting",
     "Dan from Accounting just pinged me - they're almost out of thermal paper and need a restock asap. pulled some quotes below"),
    ("toner cartridges, ~200 units", "IT",
     "IT put in a supply request for toner. we're running low across the floor printers. grabbed a few options"),
    ("ergonomic chairs x50", "Facilities",
     "facilities needs chairs for the new wing before the team moves in next month. here's what i found"),
    ("managed network switches (x8)", "IT Infrastructure",
     "got the spec from IT for the server room project. need to move on this soon, they want install done by end of month"),
    ("safety goggles, 1000 pairs", "EHS",
     "EHS flagged we need to restock goggles on the manufacturing floor. routine order, pulling quotes now"),
    ("LED panels x300", "Facilities",
     "warehouse lighting retrofit is finally getting approved - facilities needs panels ordered this week"),
    ("HVAC filters x400", "Facilities",
     "monthly filter restock for facilities. nothing urgent but want to get it ordered today"),
    ("cable management kits x100", "IT",
     "IT needs cable mgmt stuff for the infrastructure project. pretty standard, just need a vendor"),
    ("cleaning supplies, quarterly restock", "Office Ops",
     "office ops quarterly restock time. same stuff as last quarter roughly, just picking a vendor"),
    ("fire extinguishers x75", "EHS",
     "annual extinguisher replacement for EHS compliance - need to get this ordered, it's overdue"),
]

CANONICAL_REQUEST = REQUESTS[1]  # toner cartridges: generic, not compliance-adjacent
