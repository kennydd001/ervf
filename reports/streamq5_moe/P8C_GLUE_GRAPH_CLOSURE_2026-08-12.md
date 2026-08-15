# P8C — glue-only piecewise CUDA Graph: premisse gesupersedeerd

De oorspronkelijke projectie schreef 14,193 ms/token toe aan algemene “glue”
en voorspelde 5,68 ms na graph-capture. Latere componentmetingen falsificeerden
die opsplitsing: bij 4K context zat 96,6 ms in de oorspronkelijke attentionplane
en P13B reduceerde die met een exact ander algoritme tot circa 12,9 ms. P3B had
daarnaast al gemeten dat graph-capture van de expertplane geen winst gaf.

Er is geen afzonderlijke P8C-integratie uitgevoerd. Het idee is administratief
`superseded`: de numerieke voorspelling en bottleneckpremisse zijn weerlegd en
de resterende CPU-routerbarrière is in P10D fysiek negatief getest. Dit mag niet
als een positieve of fysiek geteste graph-uitkomst worden gelezen.
