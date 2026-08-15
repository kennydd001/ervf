# P-D deeltest — expertunie over speculatiediepte

De P-D-acceptatiegate kan zonder K3-target en werkend draftmodel niet worden
gemeten. Voorafgaand aan deze deeltest wordt wel het lokaal meetbare
expertunie-effect gelockt:

- gebruik de 48 officiële Qwen3-30B-A3B HERA-routertraces, alle vijf domeinen;
- dieptes `s ∈ {1,2,4,8}`;
- voor ieder mogelijk overlappend tokenvenster en iedere laag: tel de unie van
  de natuurlijke top-8 expert-IDs;
- rapporteer gemiddelde, mediaan, p95, maximum en `U(s)/(8s)` per domein;
- vergelijk met het onafhankelijke-uniforme model
  `128 * (1 - (1 - 8/128)^s)`;
- herbereken afzonderlijk de aangeleverde K3-formule voor `E=896,k=16,s=8`.

Dit is een Qwen-routecorrelatiemeting, niet K3 en niet speculative decoding.
Zonder gemeten acceptatielengte kan P-D niet slagen.
