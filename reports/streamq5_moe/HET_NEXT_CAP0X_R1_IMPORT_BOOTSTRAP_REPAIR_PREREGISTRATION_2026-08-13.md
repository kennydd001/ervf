# HET-NEXT CAP0X-R1 — import-bootstrap repair

De immutable CAP0X-poging eindigde vóór iedere device-import met
`ModuleNotFoundError: No module named 'scripts'` in beide children. Deze revisie
wijzigt uitsluitend:

1. de projectroot wordt vóór import aan `sys.path` toegevoegd;
2. alle outputpaden zijn nieuw en create-new;
3. de blokkerende `nvidia-smi`-poll in de coordinator wordt vervangen door een
   vaste diagnostische marker; proces-liveness blijft iedere 100 ms gemeten.

De frozen CAP0X-coordinator, Intel-runner, NVIDIA-runner, kernels, rondes en gates
blijven inhoudelijk ongewijzigd. Eén poging, geen retry. De claim blijft uitsluitend
een exploratieve bestaande-runner procesoverlapdiagnose.

