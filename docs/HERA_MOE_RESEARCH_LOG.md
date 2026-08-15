# HERA-MoE onderzoekslog

## 2026-08-11 — nieuwe onafhankelijke registry

HERA test de hardwarehiërarchie die zichtbaar werd in de gesloten
E2GQ-capture: een entropy-GPTQ hot tier in VRAM en een exacte BF16 cold tier in
host-RAM. De WikiTextberekening is een positieve precondition, geen
cross-domaingarantstelling.

Twee officiële FLORES-bronnen en The Stack Smol bleken zonder geconfigureerde
toegang gated. Die acquisities zijn vóór routing gestopt en append-only
behouden. De codebron is vervangen door publieke CodeXGLUE train-parquets; de
multilingual bron door een publieke, commit-gepinde FLORES-200-parquetmirror.
GSM8K en Dolly zijn uit hun gepinde publieke repositories verkregen.

P0 bevriest vijf keer 32.768 tokens en definieert hot als de union van alle
per-domein `count >=128`-sets. De 5,75-GiB-gate staat vast voordat routes worden
geopend.

## 2026-08-11 — routercapture-addenda

Poging 001 herberekende top-k uit de officiële logits en ontdekte dat BF16-ties
een tweede CUDA-`topk` andere tied indices kunnen laten kiezen. Poging 002
onderschepte daarom de werkelijke top-k-call van het officiële MoE-block en
reproduceerde alle general/E2GQ-counts exact. Alleen een overstrenge tweede
softmax-valuecheck bleef false. Addendum 002 verwijderde uitsluitend die
redundante check. Alle pogingen, routes, rapporten en hashes zijn append-only
bewaard; inputs, tierregel en gates veranderden niet.

## 2026-08-11 — P0 sluit als static-tier-negative

De definitieve officiële routes laten zien dat de WikiText-coldtier sterk
domeinspecifiek was. Hot experts per afzonderlijk domein: general 4.449, code
4.173, math 4.823, multilingual 4.317 en instruction 4.951. Hun vooraf
vastgelegde union bevat 6.081 van 6.144 laag-expertparen. Code voegt na general
alleen al 1.235 nieuwe hot experts toe; uiteindelijk blijven slechts 63 experts
cross-domain cold.

Met de bevroren diagnostische rate wordt de hotbank 6,449 GiB en de INT4-trunk
0,718 GiB: samen 7,167 GiB, **1,417 GiB boven** de 5,75-GiB-gate. Cold BF16
daalt wel tot 0,554 GiB. Alle officiële routecalls/logits zijn exact, general
reproduceert E2GQ exact en de onafhankelijke verifier sluit 15/15 controles.

Volgens de preregistratie stopt de statische count-threshold-HERA-lijn vóór
GPTQ en kwaliteit. Dit falsifieert niet alle hiërarchische of dynamische
expertcaches; een domeingeconditioneerde of budgetgestuurde hotset is een
andere hypothese en is hier niet getest. Er is geen Eureka-claim.
