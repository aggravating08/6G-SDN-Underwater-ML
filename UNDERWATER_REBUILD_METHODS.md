# Rebuilt underwater CA-SDUN methodology

This project is an analytical, event-driven ns-3 implementation of the control
and routing logic in *Cognitive Routing in Software-Defined Underwater Acoustic
Networks* (Khan et al., 2017).  It is not Bellhop/WOSS/ns-MIRACLE and it does
not claim a packet-level acoustic modem, MAC collision, interference, fading,
BER, or PER model.

## Fixed study configuration

| Item | Value |
|---|---:|
| Geometry | 2-D, 500 x 500 m² (PPT override) |
| Sensor counts | 25, 50, 75, 100 (PPT override) |
| Sensor link range | 100 m (reference paper) |
| AUV control range | 300 m (reference paper) |
| Controllers | four horizontal-track AUVs: one selected OC and three LCs; no MC |
| AUV speed | independently sampled 1.5–3.0 m/s (PPT/ML override) |
| Spectrum | 10–40 kHz, five 6-kHz channels at 13/19/25/31/37 kHz |
| OFDM parameters | 128 subcarriers, 46.875 Hz spacing, 12.4 ms CP, 21.33 ms symbol |
| PUs | two spatial mobile PUs with exponential on/off activity |
| Packet | 64 bytes, 10 flows x 20 packets = 200 packets |
| Transmit level | 150 dB re 1 uPa |
| Simulation duration | 118 s |

Each AUV has a `ConstantVelocityMobilityModel`, one fixed horizontal lane, and
one constant signed x velocity.  The four lanes and two positive/two negative
directions are randomly assigned to AUV IDs per deterministic scenario, so an
arbitrary ID is not permanently associated with a central track. Its start is
generated so it stays in the 500-m square during the 118-s run; it neither
turns nor reflects.

## Architecture

Payload forwarding is **sensor to sensor only**.  An assertion aborts if a data
route ever includes an AUV/OC.  `OC_data_hops` and
`architecture_violations` must therefore be zero.

Sensors send periodic beacons. Each AUV/LC receives only gateway beacons and
their advertised neighbours, producing four distinct localized views. A route
request uses a direct selected-OC control link when possible; otherwise it uses
a sensor gateway path scored by TD. The selected OC is the highest controller:
it computes a complete sensor-only route exclusively from its own view. If that
view cannot supply a valid route, the current packet's route setup fails; there
is no MC, global topology, or global-fallback controller.

Routes are installed per flow, revalidated per packet, and re-established for
later packets after an earlier setup failure.  A route/channel failure first
permits the best currently common idle channel for the same sensor pair, then
causes controller-assisted sensor-only recovery at the current sensor.

`gateway_control_tx` is reported as a diagnostic subset of physical
route-request/reply hops. It is never added again to total control traffic.
Thus `ROR_total = (HELLO + route requests + route replies) /
(that control total + sensor data-hop transmissions)` without double counting.

## Formulas

`D_ij = sqrt((x_i-x_j)^2 + (y_i-y_j)^2)` is used everywhere. The paper's
central controller projection in `Nhat` is retained only as a fixed mathematical
reference point `(250,250)`, not as an MC entity.

`q(z,s,T)` follows paper Eq. 9; propagation delay is `D_ij/q`.

Path loss is the dB form of Eq. 13,
`15 log10(D_ij) + (D_ij/1000) alpha(f)`, with Thorp `alpha(f)` and `f` in kHz.
The four Eq. 12 noise components are converted from dB PSD to linear power,
summed, and converted back to dB.  Link SNR is Eq. 10 in dB form.

For each 6-kHz channel, rate is the paper capacity integral
`integral log2(1+S(f)/N(f)) df`, evaluated deterministically using 60 midpoint
frequency samples.  TD is Eq. 1:
`(L_s/r_ij^ch + GD_ij) * Nhat_ij^hop`; `Nhat` is infinite for a neighbour with
zero or negative projection toward the MC.  Controller route search minimizes
the Eq. 4 LDP expression, `L_s/r_ij^ch + GD_ij + ENC`, and Eq. 5 chooses the
minimum path cost.

## Explicit implementation assumptions

The paper does not publish numerical PU on/off means, energy-detector threshold,
ordinary-link SNR eligibility threshold, beacon interval, or traffic packet
interval.  The model declares rather than attributes these assumptions:

- PU on/off means: 5 s / 5 s; the on/off law is exponential as stated in the paper.
- spatial detector and ordinary link eligibility: 3 dB SNR.
- sensor HELLO period: 10 s.
- 10 flows, 20 packets/flow, 5-s interpacket interval and 1-s flow staggering.

They are never tuned in code using PDR, E2ED, or ROR.

## ML methodology: MC-free network-performance target

The only model inputs are `x`, `y`, `local_density`, and `speed`. At selection
time, local density is the count of sensors within 300 m of an AUV. For every
generated scenario, all four candidate OCs are evaluated under identical
sensor/AUV/PU/flow conditions in the MC-free simulator. The label is selected
lexicographically: highest PDR, then lowest defined E2ED, then lowest total
ROR, then lowest AUV ID. Exactly one of four rows receives `is_oc=1`.

PDR, E2ED, ROR, route data, PU state, and topology metrics are written only to
a separate label-diagnostic ledger. They are not training columns or ML inputs.
All train/validation/test splitting is by scenario.

## Commands

```bash
./ns3 build randy
./ns3 run "scratch/randy --mode=labels --nodeCount=25 --runs=5 --baseSeed=29 --output=results/underwater_rebuild/labels.csv"
./ns3 run "scratch/randy --mode=run --nodeCount=50 --scenarioSeed=1000 --scenarioId=1000 --selectedOc=2 --output=results/underwater_rebuild/run.csv"
python3 underwater_ml_pipeline.py generate-network-labels --scenarios-per-density 500
python3 underwater_ml_pipeline.py train
```
