# Energy-Only Microgrid

This folder contains a standalone prototype for:

- day-ahead energy scheduling
- intraday rolling re-optimization
- PV uncertainty
- EV charging uncertainty
- no regulation market participation

Model assumptions in this first version:

- the system is a grid-connected PV-storage-charging microgrid
- EVs are modeled as flexible charging demand, without V2G
- day-ahead uncertainty is handled with a sample-envelope robust profile
- intraday rolling optimization uses updated profiles and applies the first control action each hour

Entry point:

```bash
python -m energy_market_mg.main
```
