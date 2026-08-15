# ST2-mini-B preregistratie — Intel host-USM Q5 met ERGV width 8

Datum: 2026-08-12  
Status bij schrijven: protocol bevroren vóór compilatie, correctness of timing
van de width-8 Intel-kernel.

## Reden voor precies deze kandidaat

ST2-mini-A gebruikte de letterlijke 256-thread bronboom met acht
workgroupbarrières. Zij bewees host-USM en Q5-bitexactheid, maar haalde slechts
9,654 GB/s aan de conservatieve p95-latentiezijde. Dit sluit die fysieke
schedule.

Er wordt niet over nieuwe widths gezocht. **Width 8** is vóór ST2 al extern
bevroren door de lokale P7/ERGV- en ERGV-C2-resultaten als de Q5 gate/up-keuze.
ST2-mini-B vertaalt uitsluitend die bekende exacte schedule naar één
Intel-subgroup per outputrij:

- 256 logische accumulatoren blijven bestaan;
- acht fysieke lanes emuleren ieder 32 virtuele accumulatoren;
- strides 128 t/m 8 gebeuren lane-lokaal;
- strides 4, 2 en 1 gebeuren met subgroup-shuffle in dezelfde
  operandvolgorde;
- 32 outputrijen delen één workgroup van 256 work-items.

## Ongewijzigde data en gates

- Intel Arc Pro 140T, geen NVIDIA-kernel of -transfer;
- dezelfde echte Qwen30 P1D Q5-bank;
- dezelfde 531-record host-USM-ring van 537.219.072 bytes
  (512,33203125 MiB), plus één downrecord voor correctness;
- `clHostMemAllocINTEL` en `clSetKernelArgMemPointerINTEL`;
- nul private weightbuffer-, write-, copy- of migratecalls;
- dezelfde normale, cancellation- en subnormal/signed-zero-inputs;
- gate, up en down tegen hetzelfde CPU-orakel;
- batch 16, 34 volledige warmupbatches en 1.000 geroteerde timed events;
- werkelijk gelezen code+scalevolume: 1.007.616 bytes/record;
- OpenCL eventprofiling en hostwallclock;
- system-wide PDH hard-page-readcontrole.

Harde gates:

```text
bitverschillen                                      0
host-USM-attest                                      pass
verborgen private weightcopy                         0
conservatieve p95-latentiezijde throughput       >= 21,63 GB/s
post-warmup Page Reads/sec                            0
```

De conservatieve snelheid is opnieuw het minimum van event-p05- en
wallclock-p05-bandbreedte. Iedere build-, exactheids-, allocatie-, page- of
throughputfail sluit **width 8 op deze Intel-backend**. Er komt binnen deze
confirmatie geen width 4/16/32-search of thresholdwijziging.

Een pass bewijst alleen een echte Intel-iGPU Q5 host-near-data component. Hij
bewijst nog geen gelijktijdige dGPU-uitvoering, cross-vendor quotientmerge,
SplitTree-laag of volledige decode.

