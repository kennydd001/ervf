# ERVF Eureka — 2026-08-12

De P7-reeks heeft een bitexacte lokale versnelling opgeleverd voor de volledige
Qwen3-30B-A3B custom decoder op de RTX PRO 2000 Blackwell Laptop GPU 8 GB:

- afgesloten test: 20,029 → 30,113 tok/s;
- 512-token greedy rollout: 15,867 → 20,915 tok/s;
- 0 verschillende CE-waarden over 1.270 validation- en 1.270 testlabels;
- dezelfde voorspellingen, routes, KV-digests en 512 rollouttokens;
- 48/48 onafhankelijke verificatiepoorten geslaagd.

De techniek, **Exact-Reduction Virtual Fusion (ERVF)**, verwerkt zestien rijen
per CUDA-block terwijl hij de oorspronkelijke 256 virtuele accumulatoren en
exact dezelfde reductieboom bewaart.

Autoritatieve documenten:

- `reports/streamq5_moe/P7_ERVF_FINAL_REPORT_2026-08-12.md`
- `reports/streamq5_moe/P7_ERVF_INDEPENDENT_VERIFICATION.md`
- `reports/streamq5_moe/P7_IDEA_AUDIT_2026-08-12.md`
- `reports/streamq5_moe/p7_ervf_independent_verification.json`

Claimgrens: bewezen lokale engineering-Eureka; nog geen wereld-SOTA- of
algemene nieuwheidsclaim zonder externe runtimebaselines en prior-artonderzoek.
