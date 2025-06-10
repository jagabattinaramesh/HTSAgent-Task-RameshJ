from agent import agent

try:
    # a bare-bones duty-only question
    print(agent.run(
        "HTS code 0101.30.00.00 cost $100 freight $0 insurance $0 weight 10 kg qty 1 units"
    ))
except Exception as e:
    print("Agent crashed with:", e)
