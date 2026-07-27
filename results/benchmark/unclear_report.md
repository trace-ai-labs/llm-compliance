# Unclear-reason full-run report (11,730 still-unclear turns)

            REFUSED_ON_P  NONCONFORMIN   HANDED_BACK     NO_ANSWER       n
overall             29.3          24.4          46.0           0.4  11,730
base                27.5          25.6          46.4           0.4   5,521
anti_adv            30.9          23.2          45.6           0.3   6,209
T1                  36.2          21.0          42.4           0.4   7,305
T2                  17.8          29.9          52.0           0.3   4,425

tier rollup (overall): substantive=53.6%  deferred=46.0%  failure=0.4%

Agreement (pairwise): raw=0.861  kappa=0.784  PABAK=0.815  AC1=0.823

Per-judge marginals:
  GLM-5.2                              REFUSE=34.1%  NONCON=24.6%  HANDED=40.9%  NO_ANS= 0.4%
  Kimi-K2.6                            REFUSE=26.5%  NONCON=20.7%  HANDED=52.3%  NO_ANS= 0.5%
  NVIDIA-Nemotron-3-Ultra-550B-A55B    REFUSE=28.9%  NONCON=26.7%  HANDED=44.1%  NO_ANS= 0.3%

Per-model (vote-share %, sorted by substantive share):
model                                     n  REFUSED_ON_P  NONCONFORMIN   HANDED_BACK     NO_ANSWER
llama-3.1-8b-instruct                 1,174          66.6          14.8          18.7           0.0
llama-3.3-70b-instruct                  879          75.5           3.9          20.2           0.5
gemini-3-flash-preview                  252           6.7          64.2          26.3           2.8
gpt-oss-120b                          1,197          46.2          24.2          29.5           0.1
inkling                                 241          22.4          48.0          29.6           0.0
gpt-5.6-luna                            245          36.6          26.3          37.1           0.0
DeepSeek-V4-Pro                         303          15.4          44.7          39.9           0.0
GLM-4.7                                 270          19.5          40.0          40.2           0.2
GLM-5.2                                 410          14.5          41.6          43.9           0.0
mistral-7b-instruct                     464           1.4          51.1          47.3           0.2
NVIDIA-Nemotron-3-Ultra-550B-A55B       382          15.8          34.7          49.5           0.0
grok-4.3                                714          40.2           7.8          52.0           0.0
gemma-4-26b                             731          16.1          31.4          52.4           0.0
Nemotron-120B-A12B                      555          16.4          30.6          47.9           5.1
qwen3.5-35b-a3b                         242          15.2          29.6          55.2           0.0
Qwen3.6 27B                             253           5.8          36.5          57.7           0.0
Kimi-K2.6                               670          16.9          24.8          58.3           0.0
GLM-5                                   471          21.3          17.3          61.4           0.0
Kimi-K2.7-Code                          378           9.2          27.9          63.0           0.0
Seed-OSS-36B-Instruct                   320           1.5          32.6          65.4           0.5
MiniMax-M2.5                            432          21.8           8.6          69.6           0.0
claude-haiku-4-5                      1,147          13.6          10.5          75.9           0.0
