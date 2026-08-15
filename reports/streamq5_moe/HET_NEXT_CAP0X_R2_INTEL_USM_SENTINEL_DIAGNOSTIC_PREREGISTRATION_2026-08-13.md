# HET-NEXT CAP0X-R2 — Intel host-USM sentinel diagnostic

CAP0X-R1 liet beide processen succesvol overlappen en NVIDIA behield een volledige
bitexacte D7 strong pass. Intel stopte vóór een devicecall omdat de eerder bewust
opgeruimde P1D-bank ontbrak. R2 vervangt alleen de Intel-child door een 4-KiB
procedurele host-USM-sentinel op dezelfde reeds bewezen OpenCL/USM-klasse.

Bevroren Intel-protocol:

- `clHostMemAllocINTEL`, 4096 bytes en alignment 4096;
- exact 1024 `uint32` inputwoorden uit de vastgelegde recurrence;
- input wordt door de CPU rechtstreeks in de host-USM-pointer geschreven;
- kernelargument 0 wordt uitsluitend met `clSetKernelArgMemPointerINTEL` gezet;
- precies 1000 kernel-launches, daarna `clFinish`;
- output wordt exact tegen een onafhankelijk CPU-orakel vergeleken;
- submit- en complete-QPC worden behouden;
- geen P1D-bank, model, Q5-kwaliteit of performanceclaim.

NVIDIA, coordinator, procesgates en claimgrens blijven gelijk aan CAP0X-R1. Eén
nieuwe poging, geen retry. Een positieve diagnose bewijst uitsluitend dat de twee
bestaande backendprocessen tijdens hetzelfde procesvenster foutvrij kunnen draaien.

