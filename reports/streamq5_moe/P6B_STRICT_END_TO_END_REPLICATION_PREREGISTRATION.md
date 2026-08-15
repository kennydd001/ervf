# STREAMQ5-MoE P6B — strikte end-to-end-replicatie

Datum: 2026-08-12. Status bij vastlegging: geen P6B-output geopend.

P6A heeft smoke, validation, test en een 512-tokenrollout technisch doorlopen,
maar de stopwatch in `decode()` startte onmiddellijk na de fysieke
host-embeddinglookup/dequantisatie. Daardoor is de P6A-snelheidsclaim niet
strikt end-to-end, ook al was de uitgesloten bewerking klein en was de output
wel van die fysieke embedding afhankelijk.

P6B is een vooraf gesloten replicatie met exact één semantische wijziging:
`wall_start` wordt vóór `embedding(token)` gezet. De volgende onderdelen blijven
ongewijzigd en worden per hash vergrendeld:

- P6A-bank, onafhankelijke bankverificatie, P1D-expertbank en modelinput;
- P0C validation/test-splits en BF16-teacherwaarden;
- P4D-calibratieroutes en domeingeconditioneerde cachekeuze;
- alle CUDA-kernels, numerieke volgorde, prompt, greedy sampling en 512 stappen;
- alle kwaliteits-, tijd-, residentie-, route-, KV- en feedbackpoorten uit P6A.

De fasevolgorde blijft smoke → validation → test → rollout. Test blijft dicht
zonder validation-pass; rollout blijft dicht zonder testkwaliteitspass. Alleen
P6B kan de definitieve end-to-end-Eureka-status dragen. P6A blijft als auditbaar
voorlopig resultaat bewaard.
