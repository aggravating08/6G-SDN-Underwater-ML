#!/usr/bin/env python3
"""v9 SVM density-trend run over frozen v8 balanced-population selections."""
from __future__ import annotations

import csv
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
V8 = ROOT / "results/underwater_rebuild/current/v8_paper_trend_balanced_pdr"
OUT = ROOT / "results/underwater_rebuild/current/v9_paper_trend_final_corrected"
RUNS = OUT / "svm_selected_candidate_runs"
EXE = ROOT / "build/scratch/ns3.38-randy-default"
NODES = (25, 50, 75, 100)
SEED_BASE = {25: (0, 4_000_029), 50: (500, 5_000_029), 75: (1000, 6_000_029), 100: (1500, 7_000_029)}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def val(row: dict[str, str], key: str) -> float:
    raw=row.get(key,"")
    return math.nan if raw in ("","NA","NaN") else float(raw)


def seed_for(nodes: int, sid: int) -> int:
    first_id, first_seed=SEED_BASE[nodes]
    return first_seed+sid-first_id


def trial(nodes: int, sid: int, oc: int) -> dict[str,str]:
    seed=seed_for(nodes,sid)
    output=RUNS / f"n{nodes}_sid{sid}_seed{seed}_oc{oc}.csv"
    if not output.exists() or sum(1 for _ in output.open()) != 2:
        output.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run([
            str(EXE),"--mode=run",f"--nodeCount={nodes}",f"--scenarioId={sid}",f"--scenarioSeed={seed}",f"--selectedOc={oc}","--runs=1",
            "--fixedControlPolicy=true","--helloSeconds=30","--routeTtlSeconds=60","--negativeRouteTtlSeconds=120",
            "--referenceUpdateAccounting=true","--aggregatedTopologyDigestAccounting=true","--topologyDigestCapacityNodes=20",
            "--bufferRouteDiscovery=true","--routeDiscoveryRetryLimit=3","--packetBufferTimeoutSeconds=60","--balancedDistanceFlows=true",
            "--relaySearchDelay=true","--relaySearchBaseMs=15","--relaySearchScaleMs=360","--relaySearchCandidateExponent=2","--relaySearchMaxMs=180",
            f"--output={output}"],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,text=True)
    rows=read(output)
    if len(rows)!=1: raise RuntimeError(f"Incomplete v9 candidate: {output}")
    return rows[0]


def mean(rows: list[dict[str,str]], key: str) -> float:
    xs=[val(r,key) for r in rows]; xs=[x for x in xs if math.isfinite(x)]
    return sum(xs)/len(xs) if xs else math.nan


def weighted_delay(rows: list[dict[str,str]]) -> float:
    pairs=[(val(r,"E2ED_ms"),val(r,"delivered_packets")) for r in rows]
    pairs=[(d,n) for d,n in pairs if math.isfinite(d) and n>0]
    return sum(d*n for d,n in pairs)/sum(n for _,n in pairs) if pairs else math.nan


def summary(rows: list[dict[str,str]], nodes: int) -> dict:
    return {"Nodes":nodes,"Scenarios":len(rows),
            "PDR %":100*sum(val(r,"delivered_packets") for r in rows)/sum(val(r,"generated_packets") for r in rows),
            "E2ED ms":weighted_delay(rows),"ROR offered":mean(rows,"ROR_generated"),"ROR transmission":mean(rows,"ROR_total"),
            "Mean S-D distance m":mean(rows,"mean_source_destination_distance_m"),"Connected-pair ratio":mean(rows,"connected_flow_pair_ratio"),
            "Delivered hop count":mean(rows,"mean_delivered_hop_count"),"Delivered path length m":mean(rows,"mean_delivered_path_length_m"),
            "Relay wait ms":mean(rows,"mean_delivered_relay_wait_ms"),"Mean candidates":mean(rows,"mean_delivered_relay_candidate_count"),
            "Topology digest packets":mean(rows,"topology_digest_packets"),"MC digest acknowledgements":mean(rows,"mc_topology_digest_ack_packets"),
            "Route control packets":mean(rows,"route_request_tx")+mean(rows,"route_reply_tx")+mean(rows,"mc_control_tx"),
            "Raw neighbor burden":mean(rows,"raw_neighbor_update_burden"),"Delivered scenario count":sum(1 for r in rows if math.isfinite(val(r,"E2ED_ms")))}


def main() -> None:
    selections=[r for r in read(V8 / "test_model_oc_predictions.csv") if r["Model"]=="SVM"]
    if len(selections)!=300: raise RuntimeError("Expected 300 frozen v8 held-out SVM predictions")
    grouped: dict[int,list[dict[str,str]]]=defaultdict(list)
    for selected in selections:
        n,sid,oc=int(selected["node_count"]),int(selected["scenario_id"]),int(selected["predicted_oc"])
        result=trial(n,sid,oc)
        if result["generated_packets"]!="200" or result["OC_data_hops"]!="0" or result["architecture_violations"]!="0":
            raise RuntimeError(f"Invariant failure n={n}, scenario={sid}, OC={oc}")
        grouped[n].append(result)
    if any(len(grouped[n])!=75 for n in NODES): raise RuntimeError("Expected 75 held-out results per density")
    raw=[r for n in NODES for r in grouped[n]]; table=[summary(grouped[n],n) for n in NODES]
    write(OUT / "svm_selected_raw_runs.csv",raw); write(OUT / "svm_density_trend.csv",table)
    pdr=[r["PDR %"] for r in table]; e2e=[r["E2ED ms"] for r in table]; ror=[r["ROR offered"] for r in table]
    gates=[
        {"Metric":"PDR %",**dict(zip(map(str,NODES),pdr)),"Expected trend":"increases then saturates (0.5 pp tolerance)","Pass/Fail":"PASS" if all(b>=a-.5 for a,b in zip(pdr,pdr[1:])) else "FAIL"},
        {"Metric":"25-node PDR", "25":pdr[0],"50":"","75":"","100":"","Expected trend":">=20%","Pass/Fail":"PASS" if pdr[0]>=20 else "FAIL"},
        {"Metric":"E2ED ms",**dict(zip(map(str,NODES),e2e)),"Expected trend":"generally decreases","Pass/Fail":"PASS" if all(b<=a for a,b in zip(e2e,e2e[1:])) else "FAIL"},
        {"Metric":"ROR offered",**dict(zip(map(str,NODES),ror)),"Expected trend":"increases","Pass/Fail":"PASS" if all(b>=a for a,b in zip(ror,ror[1:])) else "FAIL"},
        {"Metric":"ROR offered bounded",**dict(zip(map(str,NODES),ror)),"Expected trend":"around 0.4--0.5","Pass/Fail":"PASS" if all(.38<=x<=.50 for x in ror) else "FAIL"},
    ]
    write(OUT / "trend_gate.csv",gates)
    (OUT / "configuration.json").write_text(json.dumps({
        "experiment":"v9_paper_trend_final_corrected","selection":"frozen v8 SVM test selections; density-figure protocol sensitivity only; no v9 ML relabel/retraining",
        "balanced_source_destination_rule_m":"300--450; static connected pair preferred","hello_update_interval_seconds":30,"route_ttl_seconds":60,
        "relay_search_formula_ms":"min(15 + 360/max(1,candidate_count^2), 180)","topology_digest_capacity_nodes":20,
        "mc_digest_acknowledgement":"one control acknowledgement per nonempty LC region per update interval",
        "ror_offered":"control/(control+200 generated packets)","ror_transmission":"control/(control+realized data hops)"},indent=2)+"\n")
    print("V9 SVM DENSITY TREND"); [print(r) for r in table]; print("V9 GATES"); [print(r) for r in gates]


if __name__=="__main__": main()
