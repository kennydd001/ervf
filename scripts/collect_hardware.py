from moe_lab.hardware import collect_hardware
from moe_lab.reporting import envelope, write_json


if __name__ == "__main__":
    report = envelope("hardware_baseline", collect_hardware())
    path = write_json("hardware.json", report)
    print(path)

