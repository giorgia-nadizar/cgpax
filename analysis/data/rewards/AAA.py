import pandas as pd

df = pd.read_csv("hopper_cgp_mu+lambda-ga_reward_0_999.csv")
sums = df[["healthy", "ctrl", "forward", "total"]].sum()
print(sums)