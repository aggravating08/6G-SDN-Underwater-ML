/*
 * Analytical hierarchical underwater-SDN experiment.
 *
 * This is an ns-3 event-driven analytical model, not a UAN PHY/MAC modem
 * simulation.  It deliberately excludes Bellhop/WOSS/ns-MIRACLE, collisions,
 * BER/PER and payload forwarding by AUVs.  The project-specific configuration is
 * 500 x 500 m 2-D geometry, 25/50/75/100 sensors, four AUVs, one
 * ML-selected OC, and one surface MC. In partner-style simple mode the MC is
 * not used as a global route fallback.  The remaining AUVs collect local
 * state as LCs.  AUVs/controllers never forward user payload.
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;
NS_LOG_COMPONENT_DEFINE("CaSdunUnderwater");

namespace {

// PPT-fixed study geometry and ML-mobility override.
constexpr double kArea = 500.0;
constexpr uint32_t kNumAuv = 4;
constexpr double kSensorRange = 100.0;
constexpr double kAuvControlRange = 300.0;
// The MC is a surface controller whose horizontal reference projection is the
// centre of the study area.  It participates in control traffic only.
constexpr double kControlReferenceX = 250.0, kControlReferenceY = 250.0;
constexpr uint32_t kPacketBytes = 64;
constexpr uint32_t kPacketBits = kPacketBytes * 8;
constexpr uint32_t kFlows = 10, kPacketsPerFlow = 20;
constexpr double kPacketInterval = 5.0; // traffic assumption; paper gives no interval
constexpr double kFlowStagger = 1.0;
constexpr double kFlowStart = 8.0;
constexpr double kSimulationTime = 118.0;
constexpr double kHelloPeriod = 60.0; // base periodic beacon assumption; paper gives no interval
constexpr double kTxPowerDb = 150.0;  // dB re 1 uPa (paper)
constexpr double kLinkSnrThresholdDb = 3.0; // explicit analytical eligibility assumption
constexpr double kControlBytes = 30.0;
constexpr double kControllerLookupMs = 3.0;
// Installed sensor-only routes are revalidated on use and invalidated on a
// physical link failure.  Their cache lifetime is deliberately independent of
// the 60-s HELLO refresh interval.
constexpr double kRouteTtl = 120.0;
// A confirmed MC no-path result is retained only briefly.  This suppresses
// duplicate reactive control for a currently unreachable sensor pair without
// permanently blacklisting it as AUV position/control state changes.
constexpr double kNegativeRouteTtl = 30.0;
constexpr double kEps = std::numeric_limits<double>::epsilon();

// Runtime policy overrides are used only by separately named protocol
// experiments.  The default remains the current density-adaptive policy, so
// previously generated ledgers remain reproducible.  They never affect ML
// labels, traffic generation, or payload forwarding.
bool gUseFixedControlPolicy = false;
double gFixedHelloPeriod = kHelloPeriod;
double gFixedNegativeRouteTtl = kNegativeRouteTtl;
double gFixedRouteTtl = kRouteTtl;
// Separate reference-style accounting mode.  It represents the source
// paper's periodic sensor-neighbour state exchange and controller summaries
// explicitly, without changing payload routing or ML selection.
bool gUseReferenceUpdateAccounting = false;
bool gBufferRouteDiscovery = false;
uint32_t gRouteDiscoveryRetryLimit = 0;
double gPacketBufferTimeoutSeconds = 0.;
// Simple partner-style option: when true, LC/selected-OC failure does not get
// rescued by MC global routing.  This makes OC choice affect PDR, delay and ROR
// more strongly while preserving sensor-only payload forwarding.
bool gDisableMcFallback = false;
bool gLongDistanceFlows = false;
constexpr double kLongDistanceFlowMinM = 424.3;
bool gBalancedDistanceFlows = false;
constexpr double kBalancedFlowMinM = 300.;
constexpr double kBalancedFlowMaxM = 450.;
// The clean aligned v10 experiment uses the same moderate-long endpoint
// distance rule, but accepts only statically connected sensor-only pairs in a
// comparable hop band.  This is traffic-population design, not a routing
// shortcut: data forwarding still sees only the ordinary dynamic eligibility
// checks at run time.
bool gHopBalancedDistanceFlows = false;
// Optional v5 reference-trend mechanism.  This is an analytical local
// next-hop discovery wait, not an error model or an outcome-dependent delay.
// It depends solely on the feasible positive-progress neighbours visible at
// the forwarding sensor when the hop is attempted.
bool gUseRelaySearchDelay = false;
double gRelaySearchBaseMs = 5.;
double gRelaySearchScaleMs = 180.;
double gRelaySearchMaxMs = 80.;
double gRelaySearchCandidateExponent = 1.;
// Accounting-only protocol setting for aggregated SDN topology digests.  The
// raw neighbour reception-equivalent burden remains available as a separate
// diagnostic; only this compressed count contributes to the reported control
// transmission total when the factor is below one.
double gTopologyUpdateCompressionFactor = 1.;
// v7 protocol accounting: local neighbour discovery is summarized by the
// three LC service regions rather than charged as one network-control packet
// per sensor or per neighbour reception.  This changes accounting only, not
// the routing database, link eligibility, payload path, or timing.
bool gUseAggregatedTopologyDigestAccounting = false;
uint32_t gTopologyDigestCapacityNodes = 25;

// The PPT specifies a 25-kHz AUV operating frequency and the 1--40 kHz
// acoustic band.  There is one fixed acoustic link model: no cognitive
// channels, primary users, spectrum sensing, or idle-channel selection.
constexpr double kOperatingFrequencyKhz = 25.0;
constexpr double kBandLowKhz = 1.0, kBandHighKhz = 40.0;
constexpr double kAcousticBandwidthKhz = kBandHighKhz - kBandLowKhz;

struct Pos { double x = 0, y = 0; };
struct Sensor { Pos p; };
struct Auv { Pos start; double speed = 0, direction = 1; double laneY = 0; };
struct Flow { uint32_t source = 0, destination = 0; };

static double Distance(const Pos& a, const Pos& b) {
  const double dx = a.x - b.x, dy = a.y - b.y;
  return std::sqrt(dx * dx + dy * dy);
}

// Equation (9), with z in km and T = temperature(C)/10.
static double SoundSpeed(double depthM = 50., double salinity = 35., double tempC = 15.) {
  const double z = depthM / 1000.0;
  const double T = tempC / 10.0;
  return 1449.05 + 45.7*T - 5.21*T*T + 0.23*T*T*T +
         (1.333 - 0.126*T + 0.009*T*T) * (salinity - 35.) + 16.3*z + 0.18*z*z;
}

// Equation (13): f in kHz, alpha in dB/km.
static double ThorpDbPerKm(double fKhz) {
  const double f2 = fKhz * fKhz;
  return 0.11*f2/(1.+f2) + 44.*f2/(4100.+f2) + 2.75e-4*f2 + 0.003;
}

// Eq. (13) written in dB: 10*k*log10(d) + d(km)*alpha(f), k=1.5.
static double PathLossDb(double distanceM, double fKhz) {
  const double d = std::max(distanceM, 1.0);
  return 15.0 * std::log10(d) + (d / 1000.0) * ThorpDbPerKm(fKhz);
}

// Eq. (12).  Components are power-summed, which is the physical
// interpretation of the paper's stated superposition of dB PSD components.
static double AmbientNoiseDb(double fKhz, double shipping = 0.5, double wind = 0.0) {
  const double nt = 17. - 30.*std::log10(fKhz);
  const double ns = 40. + 20.*(shipping - .5) + 26.*std::log10(fKhz)
                    - 60.*std::log10(fKhz + .03);
  const double nw = 50. + 7.5*std::sqrt(wind) + 20.*std::log10(fKhz)
                    - 40.*std::log10(fKhz + .4);
  const double nth = -15. + 20.*std::log10(fKhz);
  return 10.*std::log10(std::pow(10., nt/10.) + std::pow(10., ns/10.) +
                         std::pow(10., nw/10.) + std::pow(10., nth/10.));
}

// Equation (10) in dB over the configured fixed acoustic band.
static double SnrDb(double distanceM, double fKhz) {
  return kTxPowerDb - PathLossDb(distanceM, fKhz) - AmbientNoiseDb(fKhz)
         - 10.*std::log10(kAcousticBandwidthKhz * 1000.);
}

static double PropagationSeconds(double distanceM) { return distanceM / SoundSpeed(); }

// PPT capacity integral r = integral_{1 kHz}^{40 kHz} log2(1+S(f)/N(f)) df,
// evaluated by deterministic midpoint quadrature.  The 25-kHz value is the
// fixed operating/reference frequency; it is not one of several channels.
static double AcousticRateBps(double distanceM) {
  const double low = kBandLowKhz;
  constexpr int samples = 60;
  const double deltaKhz = kAcousticBandwidthKhz / samples;
  double bitsPerSecond = 0.0;
  for (int i = 0; i < samples; ++i) {
    const double f = low + (i + .5) * deltaKhz;
    const double gamma = std::pow(10., SnrDb(distanceM, f) / 10.);
    bitsPerSecond += std::log2(1. + gamma) * deltaKhz * 1000.;
  }
  return bitsPerSecond;
}

static uint64_t FnvMix(uint64_t h, uint64_t value) {
  h ^= value; return h * 1099511628211ULL;
}
static uint64_t Quant(double x) { return static_cast<uint64_t>(std::llround(x * 1000.)); }
static std::string Hex(uint64_t x) { std::ostringstream s; s << std::hex << x; return s.str(); }

struct ScenarioRecord { uint32_t id = 0, seed = 0; };

// A manifest is a simple two-column CSV (scenario_id,scenario_seed).  It is
// used only to batch already accepted deterministic scenarios; it does not
// alter generation, routing, or RNG streams within a scenario.
static std::vector<ScenarioRecord> ReadScenarioManifest(const std::string& path) {
  if (path.empty()) return {};
  std::ifstream input(path);
  if (!input) NS_ABORT_MSG("Cannot open scenario manifest: " << path);
  std::vector<ScenarioRecord> records;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty() || !std::isdigit(static_cast<unsigned char>(line.front()))) continue;
    const auto comma = line.find(',');
    if (comma == std::string::npos) NS_ABORT_MSG("Malformed scenario manifest row: " << line);
    records.push_back({static_cast<uint32_t>(std::stoul(line.substr(0, comma))),
                       static_cast<uint32_t>(std::stoul(line.substr(comma + 1)))});
  }
  if (records.empty()) NS_ABORT_MSG("Scenario manifest contains no records: " << path);
  return records;
}

struct Edge {
  bool usable = false;
  double distanceM = 0, snrDb = -std::numeric_limits<double>::infinity(), rateBps = 0;
};
struct StaticSensorLink {
  double distanceM = 0.;
  double snrDb = -std::numeric_limits<double>::infinity();
  double rateBps = 0.;
};

// Diagnostics computed from the initial sensor-only graph.  The path is used
// solely to accept comparable traffic pairs and report their difficulty; it
// is never installed as a forwarding route or exposed to ML as an input.
struct StaticPathDiagnostic {
  bool connected = false;
  uint32_t hops = 0;
  double pathLengthM = std::numeric_limits<double>::quiet_NaN();
  double meanFeasibleRelayCandidates = std::numeric_limits<double>::quiet_NaN();
};

struct Route {
  bool valid = false;
  std::vector<uint32_t> nodes;
  double cost = std::numeric_limits<double>::infinity();
  uint64_t hash = 0;
};

struct View {
  std::set<uint32_t> knownSensors;
  std::map<std::pair<uint32_t,uint32_t>, Edge> edges;
  double updatedAt = -std::numeric_limits<double>::infinity();
};

struct PreRoutingOcFeature {
  double sourceDistanceM = 0., destinationDistanceM = 0.;
  uint32_t sensorsInRange = 0, gatewaysInRange = 0;
  double sourceCoveredFraction = 0., destinationCoveredFraction = 0.;
  double localPathCoverage = 0., averageLocalLinkQuality = 0.;
  double estimatedHopCount = 0.;
};

struct FlowState {
  Route installed;
  double expiresAt = -1;
  uint32_t generated = 0, delivered = 0, setupAttempts = 0, setupSuccesses = 0,
           setupFailures = 0, droppedNoRoute = 0, reestablishments = 0;
};

struct CachedRoute {
  Route route;
  double expiresAt = -1.;
};

struct NegativeRoute {
  double expiresAt = -1.;
};

struct Metrics {
  uint32_t generated = 0, delivered = 0, dataHops = 0, ocDataHops = 0,
           architectureViolations = 0, helloTx = 0, routeRequestTx = 0,
           routeReplyTx = 0, mcControlTx = 0, gatewayControlTx = 0,
           neighborUpdateTx = 0, lcTopologyUpdateTx = 0, mcTopologyUpdateTx = 0,
           topologyDigestTx = 0,
           mcTopologyDigestAckTx = 0,
           controllerRequests = 0, lcCacheHits = 0, lcCacheMisses = 0,
           ocCacheHits = 0, ocCacheMisses = 0, ocRouteComputations = 0,
           installedRouteHits = 0,
           ocRouteSuccesses = 0, ocRouteFailures = 0, mcFallbacks = 0,
           mcRouteComputations = 0, mcRouteSuccesses = 0, mcRouteFailures = 0,
           routeRecoveries = 0, recoverySuccesses = 0,
           recoveryFailures = 0, linkInvalidations = 0, controlUnreachable = 0,
           noPhysicalPath = 0, greedyDeadEnd = 0, routingFailureWithPath = 0,
           finalNoPath = 0, distanceFailures = 0, snrFailures = 0, loopFailures = 0,
           failedRouteAttempts = 0, negativeCacheHits = 0,
           pendingRouteSuppressions = 0;
  uint32_t routeDiscoveryBufferedRetries = 0, routeDiscoveryTimeoutDrops = 0;
  double delayMsTotal = 0, propagationMsTotal = 0, serializationMsTotal = 0,
         controllerWaitMsTotal = 0, deliveredPathLengthMTotal = 0.,
         deliveredRelayCandidateCountTotal = 0., deliveredRelayWaitMsTotal = 0.;
  uint32_t deliveredHopCountTotal = 0;
};

static double CompressedTopologyUpdates(const Metrics& m) {
  if (gUseAggregatedTopologyDigestAccounting)
    return static_cast<double>(m.topologyDigestTx + m.mcTopologyDigestAckTx);
  // A sensor beacon is one broadcast transmission, even though several
  // neighbours may receive it.  neighborUpdateTx records those receptions as
  // a topology diagnostic only; charging it again here would double-count the
  // same control packet after helloTx already counted its sender broadcast.
  return static_cast<double>(m.lcTopologyUpdateTx) + static_cast<double>(m.mcTopologyUpdateTx);
}

static double TotalControlPackets(const Metrics& m) {
  // Paper-faithful packet accounting: helloTx is exactly one broadcast control
  // packet per sensor sender.  Its neighbour receptions are retained in
  // neighborUpdateTx for diagnostics, but are not separate transmissions.
  // LC and MC values are actual controller-summary transmissions.
  const double periodic = gUseAggregatedTopologyDigestAccounting
      ? CompressedTopologyUpdates(m)
      : static_cast<double>(m.helloTx) + CompressedTopologyUpdates(m);
  return periodic +
         static_cast<double>(m.routeRequestTx) + static_cast<double>(m.routeReplyTx) +
         static_cast<double>(m.mcControlTx);
}

class Scenario {
public:
  Scenario(uint32_t seed, uint32_t sensors, uint32_t scenarioId, uint32_t selectedOc,
           bool trace)
    : m_seed(seed), m_n(sensors), m_scenarioId(scenarioId), m_selectedOc(selectedOc), m_trace(trace) {
    Generate();
    InstallMobility();
  }

  ~Scenario() = default;

  void Run() {
    Simulator::Schedule(Seconds(0.), &Scenario::Beacon, this);
    for (uint32_t f = 0; f < kFlows; ++f) {
      for (uint32_t p = 0; p < kPacketsPerFlow; ++p) {
        const double t = kFlowStart + f * kFlowStagger + p * kPacketInterval;
        Simulator::Schedule(Seconds(t), &Scenario::LaunchPacket, this, f, p);
      }
    }
    Simulator::Stop(Seconds(kSimulationTime));
    Simulator::Run();
  }

  const Metrics& GetMetrics() const { return m_metrics; }
  const std::vector<FlowState>& GetFlowStates() const { return m_flowStates; }
  uint64_t TopologyHash() const { return m_topologyHash; }
  uint64_t AcousticStateHash() const { return m_acousticStateHash; }
  uint64_t LastRouteHash() const { return m_lastRouteHash; }
  uint32_t SelectedOc() const { return m_selectedOc; }
  uint32_t Seed() const { return m_seed; }
  uint32_t NodeCount() const { return m_n; }
  const std::vector<Auv>& Auvs() const { return m_auvs; }
  std::vector<double> LocalDensities() const {
    std::vector<double> d(kNumAuv, 0.0);
    for (uint32_t a = 0; a < kNumAuv; ++a) {
      const Pos p = AuvPosition(a, 0.0);
      for (const auto& s : m_sensors) if (Distance(p, s.p) <= kAuvControlRange) d[a] += 1.;
    }
    return d;
  }
  const std::vector<Flow>& Flows() const { return m_flows; }
  double MeanSourceDestinationDistance() const {
    if (m_flows.empty()) return 0.;
    double total=0.;
    for (const auto& flow : m_flows) total += Distance(m_sensors[flow.source].p, m_sensors[flow.destination].p);
    return total / m_flows.size();
  }
  double ConnectedFlowPairRatio() const {
    if (m_flows.empty()) return 0.;
    uint32_t connected=0;
    for (const auto& flow : m_flows)
      if (StaticSensorPathExists(flow.source, flow.destination)) ++connected;
    return static_cast<double>(connected) / m_flows.size();
  }
  double MeanStaticShortestHops() const {
    if (m_flows.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.;
    for (const auto& flow : m_flows) sum += StaticPathDiagnostics(flow.source, flow.destination).hops;
    return sum / m_flows.size();
  }
  double MeanStaticShortestPathLengthM() const {
    if (m_flows.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.; uint32_t count = 0;
    for (const auto& flow : m_flows) {
      const auto diagnostic = StaticPathDiagnostics(flow.source, flow.destination);
      if (std::isfinite(diagnostic.pathLengthM)) { sum += diagnostic.pathLengthM; ++count; }
    }
    return count ? sum / count : std::numeric_limits<double>::quiet_NaN();
  }
  double MeanStaticFeasibleRelayCandidates() const {
    if (m_flows.empty()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.; uint32_t count = 0;
    for (const auto& flow : m_flows) {
      const auto diagnostic = StaticPathDiagnostics(flow.source, flow.destination);
      if (std::isfinite(diagnostic.meanFeasibleRelayCandidates)) {
        sum += diagnostic.meanFeasibleRelayCandidates; ++count;
      }
    }
    return count ? sum / count : std::numeric_limits<double>::quiet_NaN();
  }
  const std::vector<Sensor>& Sensors() const { return m_sensors; }
  std::vector<PreRoutingOcFeature> PreRoutingOcFeatures() const {
    std::vector<PreRoutingOcFeature> output(kNumAuv);
    for (uint32_t a=0; a<kNumAuv; ++a) {
      const Pos auv=AuvPosition(a,0.);
      const View view=LocalViewForAuv(a,0.);
      auto& f=output[a];
      uint32_t sourceCovered=0, destinationCovered=0, localPaths=0;
      for (const auto& flow : m_flows) {
        const Pos& source=m_sensors[flow.source].p;
        const Pos& destination=m_sensors[flow.destination].p;
        f.sourceDistanceM += Distance(auv,source);
        f.destinationDistanceM += Distance(auv,destination);
        if (SensorAuvEdge(flow.source,a,0.).usable) ++sourceCovered;
        if (SensorAuvEdge(flow.destination,a,0.).usable) ++destinationCovered;
        if (ShortestRoute(flow.source,flow.destination,0.,view,{}).valid) ++localPaths;
      }
      for (uint32_t sensor=0;sensor<m_n;++sensor)
        if (SensorAuvEdge(sensor,a,0.).usable) ++f.sensorsInRange;
      // In this architecture every sensor in range can serve as an LC gateway.
      f.gatewaysInRange=f.sensorsInRange;
      for (const auto& [ignored,e] : view.edges) {
        (void)ignored;
        f.averageLocalLinkQuality += std::clamp((e.snrDb-kLinkSnrThresholdDb)/20.,0.,1.);
      }
      if (!view.edges.empty()) f.averageLocalLinkQuality/=view.edges.size();
      const double flowCount=static_cast<double>(m_flows.size());
      f.sourceDistanceM/=flowCount; f.destinationDistanceM/=flowCount;
      f.sourceCoveredFraction=sourceCovered/flowCount;
      f.destinationCoveredFraction=destinationCovered/flowCount;
      f.localPathCoverage=localPaths/flowCount;
      f.estimatedHopCount=f.destinationDistanceM/kSensorRange;
    }
    return output;
  }

private:
  uint32_t m_seed, m_n, m_scenarioId, m_selectedOc;
  bool m_trace;
  std::vector<Sensor> m_sensors;
  std::vector<Auv> m_auvs;
  std::vector<Flow> m_flows;
  std::vector<StaticSensorLink> m_staticSensorLinks;
  std::vector<FlowState> m_flowStates;
  std::array<View, kNumAuv> m_views;
  View m_mcView;
  // LCs retain routes returned by the selected OC/MC for a requesting sensor;
  // the selected OC retains its own valid local/global decisions.  Entries are
  // always revalidated before use and expire independently of HELLO refresh.
  std::array<std::map<std::pair<uint32_t,uint32_t>, CachedRoute>, kNumAuv> m_lcRouteCaches;
  std::map<std::pair<uint32_t,uint32_t>, CachedRoute> m_ocRouteCache;
  // Scenario has exactly one selected OC, so these per-pair entries are also
  // implicitly scoped to that OC.  They are control-plane state only.
  std::map<std::pair<uint32_t,uint32_t>, NegativeRoute> m_negativeRouteCache;
  std::set<std::pair<uint32_t,uint32_t>> m_pendingRouteRequests;
  Metrics m_metrics;
  NodeContainer m_auvNodes;
  std::vector<Ptr<ConstantVelocityMobilityModel>> m_auvMobility;
  uint64_t m_topologyHash = 0, m_acousticStateHash = 0, m_lastRouteHash = 0;
  uint64_t m_lastSensorStateSignature = 0;
  bool m_hasSensorStateSignature = false;

  double HelloPeriod() const {
    if (gUseFixedControlPolicy) return gFixedHelloPeriod;
    switch (m_n) {
      case 25: return 60.;
      case 50: return 75.;
      case 75: return 90.;
      case 100: return 120.;
      default: return kHelloPeriod;
    }
  }

  double NegativeRouteTtl() const {
    if (gUseFixedControlPolicy) return gFixedNegativeRouteTtl;
    if (m_n == 25) return 60.;
    if (m_n == 50) return 45.;
    return kNegativeRouteTtl;
  }

  double RouteTtl() const {
    return gUseFixedControlPolicy ? gFixedRouteTtl : kRouteTtl;
  }

  double RouteRetryBackoffSeconds(uint32_t retryOrdinal) const {
    // One analytical acoustic control round at the configured control range,
    // with bounded exponential backoff.  This is independent of node count.
    const double controlRound = 2. * PropagationSeconds(kAuvControlRange) +
                                kControllerLookupMs / 1000.;
    return controlRound * std::pow(2., static_cast<double>(retryOrdinal));
  }

  bool UsesDenseSummaryUpdates() const {
    return !gUseFixedControlPolicy && (m_n == 75 || m_n == 100);
  }

  uint64_t SensorStateSignature() const {
    // Sensor locations and their analytical link qualities are static in this
    // model.  The signature makes delta suppression explicit and would change
    // if a future model updates a sensor-neighbour/link-state entry.
    uint64_t h = 1469598103934665603ULL;
    for (const auto& edge : m_staticSensorLinks) {
      h = FnvMix(h, Quant(edge.distanceM));
      h = FnvMix(h, std::isfinite(edge.snrDb) ? Quant(edge.snrDb)
                                               : std::numeric_limits<uint64_t>::max());
    }
    return h;
  }

  void Generate() {
    std::mt19937_64 rng(m_seed);
    std::uniform_real_distribution<double> coord(0., kArea);
    std::uniform_real_distribution<double> speed(1.5, 3.0);
    std::uniform_int_distribution<uint32_t> sensorId(0, m_n - 1);
    m_sensors.resize(m_n);
    for (auto& s : m_sensors) s.p = {coord(rng), coord(rng)};
    m_staticSensorLinks.resize(static_cast<size_t>(m_n) * m_n);
    for (uint32_t i=0;i<m_n;++i) for (uint32_t j=0;j<m_n;++j) {
      auto& link=m_staticSensorLinks[static_cast<size_t>(i)*m_n+j];
      link.distanceM=Distance(m_sensors[i].p,m_sensors[j].p);
      if(i==j || link.distanceM>kSensorRange) continue;
      link.snrDb=SnrDb(link.distanceM,kOperatingFrequencyKhz);
      link.rateBps=AcousticRateBps(link.distanceM);
    }

    // Fixed straight horizontal tracks.  Starts are chosen so no AUV exits the
    // area during the complete 118-s run and its configured speed is preserved.
    std::array<double,4> lanes{{75., 175., 325., 425.}};
    // Candidate IDs have no physical meaning.  Randomly assigning the same
    // fixed horizontal tracks (and two directions in each sense) prevents a
    // central lane from being permanently tied to OC1/OC2 in every scenario.
    std::array<double,4> directions{{1., 1., -1., -1.}};
    std::shuffle(lanes.begin(), lanes.end(), rng);
    std::shuffle(directions.begin(), directions.end(), rng);
    m_auvs.resize(kNumAuv);
    for (uint32_t a = 0; a < kNumAuv; ++a) {
      const double v = speed(rng), travel = v * kSimulationTime;
      const double sign = directions[a];
      std::uniform_real_distribution<double> start(sign > 0 ? 0. : travel,
                                                    sign > 0 ? kArea-travel : kArea);
      m_auvs[a] = {{start(rng), lanes[a]}, v, sign, lanes[a]};
    }
    m_flows.resize(kFlows);
    if (gHopBalancedDistanceFlows) {
      std::vector<Flow> strictEligible, fallbackEligible;
      const uint32_t strictMinHops = 5, maxHops = 8;
      for (uint32_t source = 0; source < m_n; ++source) {
        for (uint32_t destination = 0; destination < m_n; ++destination) {
          if (source == destination) continue;
          const double distance = Distance(m_sensors[source].p, m_sensors[destination].p);
          if (distance < kBalancedFlowMinM || distance > kBalancedFlowMaxM) continue;
          const auto diagnostic = StaticPathDiagnostics(source, destination);
          if (!diagnostic.connected || diagnostic.hops > maxHops) continue;
          if (diagnostic.hops >= strictMinHops) strictEligible.push_back({source, destination});
          if (m_n == 25 && diagnostic.hops >= 4) fallbackEligible.push_back({source, destination});
        }
      }
      // The 4--8 fallback is used only when this 25-node topology has no
      // 5--8-hop moderate-long connected pair.  No disconnected or short
      // pair is ever admitted.
      const auto& eligible = !strictEligible.empty() ? strictEligible : fallbackEligible;
      if (eligible.empty()) {
        NS_ABORT_MSG("Hop-balanced traffic selection found no connected 300--450 m pair in the required hop band");
      }
      std::vector<Flow> shuffled = eligible;
      std::shuffle(shuffled.begin(), shuffled.end(), rng);
      for (uint32_t f = 0; f < kFlows; ++f) m_flows[f] = shuffled[f % shuffled.size()];
    } else if (gBalancedDistanceFlows) {
      std::vector<Flow> inRange, connectedInRange;
      for (uint32_t source=0; source<m_n; ++source) {
        for (uint32_t destination=0; destination<m_n; ++destination) {
          if (source==destination) continue;
          const double distance=Distance(m_sensors[source].p,m_sensors[destination].p);
          if (distance < kBalancedFlowMinM || distance > kBalancedFlowMaxM) continue;
          const Flow pair{source,destination};
          inRange.push_back(pair);
          if (StaticSensorPathExists(source,destination)) connectedInRange.push_back(pair);
        }
      }
      if (inRange.empty())
        NS_ABORT_MSG("Balanced traffic selection found no 300--450 m sensor pair");
      // Prefer physically connected moderate-long pairs.  If the static graph
      // has no such pair, retain an in-range pair and let routing fail without
      // fabricating a path.
      auto& eligible=connectedInRange.empty() ? inRange : connectedInRange;
      std::shuffle(eligible.begin(),eligible.end(),rng);
      for (uint32_t f=0;f<kFlows;++f) m_flows[f]=eligible[f % eligible.size()];
    } else if (gLongDistanceFlows) {
      std::vector<Flow> eligible;
      for (uint32_t source=0; source<m_n; ++source) {
        for (uint32_t destination=0; destination<m_n; ++destination) {
          if (source != destination &&
              Distance(m_sensors[source].p, m_sensors[destination].p) >= kLongDistanceFlowMinM)
            eligible.push_back({source, destination});
        }
      }
      if (eligible.empty()) {
        NS_ABORT_MSG("Long-distance traffic selection found no sensor pair at least 424.3 m apart");
      }
      std::shuffle(eligible.begin(), eligible.end(), rng);
      // A scenario uses distinct eligible pairs whenever at least ten exist;
      // wrapping is deterministic and preserves the same long-distance rule.
      for (uint32_t f=0; f<kFlows; ++f) m_flows[f]=eligible[f % eligible.size()];
    } else {
      for (auto& f : m_flows) {
        f.source = sensorId(rng); do { f.destination = sensorId(rng); } while (f.destination == f.source);
      }
    }
    m_flowStates.resize(kFlows);

    uint64_t th = 1469598103934665603ULL;
    for (const auto& s : m_sensors) { th=FnvMix(th,Quant(s.p.x)); th=FnvMix(th,Quant(s.p.y)); }
    for (const auto& a : m_auvs) { th=FnvMix(th,Quant(a.start.x)); th=FnvMix(th,Quant(a.start.y)); th=FnvMix(th,Quant(a.speed)); }
    for (const auto& f : m_flows) { th=FnvMix(th,f.source); th=FnvMix(th,f.destination); }
    uint64_t acoustic = 1469598103934665603ULL;
    acoustic=FnvMix(acoustic,Quant(kOperatingFrequencyKhz));
    acoustic=FnvMix(acoustic,Quant(kBandLowKhz));
    acoustic=FnvMix(acoustic,Quant(kBandHighKhz));
    m_topologyHash = th; m_acousticStateHash = acoustic;
  }

  void InstallMobility() {
    m_auvNodes.Create(kNumAuv);
    MobilityHelper mobility;
    mobility.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    mobility.Install(m_auvNodes);
    m_auvMobility.resize(kNumAuv);
    for (uint32_t a = 0; a < kNumAuv; ++a) {
      auto model = m_auvNodes.Get(a)->GetObject<ConstantVelocityMobilityModel>();
      model->SetPosition(Vector(m_auvs[a].start.x, m_auvs[a].start.y, 0.));
      model->SetVelocity(Vector(m_auvs[a].direction * m_auvs[a].speed, 0., 0.));
      m_auvMobility[a] = model;
    }
  }

  Pos AuvPosition(uint32_t a, double /*time*/) const {
    const Vector p = m_auvMobility.at(a)->GetPosition();
    return {p.x, p.y};
  }
  Edge SensorEdge(uint32_t i, uint32_t j, double /*time*/) const {
    Edge best;
    const auto& fixed=m_staticSensorLinks[static_cast<size_t>(i)*m_n+j];
    const double d = fixed.distanceM;
    best.distanceM = d;
    if (d <= kSensorRange && fixed.snrDb >= kLinkSnrThresholdDb)
      best = {true,d,fixed.snrDb,fixed.rateBps};
    return best;
  }
  bool StaticSensorPathExists(uint32_t source,uint32_t destination) const {
    std::vector<bool> seen(m_n,false);
    std::queue<uint32_t> queue;
    seen[source]=true; queue.push(source);
    while(!queue.empty()) {
      const uint32_t current=queue.front(); queue.pop();
      if(current==destination) return true;
      for(uint32_t next=0;next<m_n;++next) {
        if(seen[next] || next==current) continue;
        const auto& edge=m_staticSensorLinks[static_cast<size_t>(current)*m_n+next];
        if(edge.distanceM<=kSensorRange && edge.snrDb>=kLinkSnrThresholdDb) {
          seen[next]=true; queue.push(next);
        }
      }
    }
    return false;
  }
  StaticPathDiagnostic StaticPathDiagnostics(uint32_t source, uint32_t destination) const {
    StaticPathDiagnostic out;
    if (source >= m_n || destination >= m_n) return out;
    const uint32_t noParent = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> parent(m_n, noParent);
    std::queue<uint32_t> queue;
    parent[source] = source;
    queue.push(source);
    while (!queue.empty()) {
      const uint32_t current = queue.front(); queue.pop();
      if (current == destination) break;
      for (uint32_t next = 0; next < m_n; ++next) {
        if (parent[next] != noParent || next == current) continue;
        const auto& edge = m_staticSensorLinks[static_cast<size_t>(current) * m_n + next];
        if (edge.distanceM <= kSensorRange && edge.snrDb >= kLinkSnrThresholdDb) {
          parent[next] = current;
          queue.push(next);
        }
      }
    }
    if (parent[destination] == noParent) return out;
    std::vector<uint32_t> path;
    for (uint32_t current = destination;; current = parent[current]) {
      path.push_back(current);
      if (current == source) break;
    }
    std::reverse(path.begin(), path.end());
    out.connected = true;
    out.hops = static_cast<uint32_t>(path.size() - 1);
    out.pathLengthM = 0.;
    for (size_t i = 1; i < path.size(); ++i) {
      out.pathLengthM += m_staticSensorLinks[static_cast<size_t>(path[i - 1]) * m_n + path[i]].distanceM;
    }
    // Along the deterministic minimum-hop diagnostic path, count neighbours
    // that are physically eligible and strictly closer to the destination.
    // This is a local relay-availability diagnostic, not a controller hint.
    if (out.hops > 0) {
      double candidates = 0.;
      for (size_t index = 0; index + 1 < path.size(); ++index) {
        const uint32_t current = path[index];
        const double currentDistance = Distance(m_sensors[current].p, m_sensors[destination].p);
        for (uint32_t next = 0; next < m_n; ++next) {
          if (next == current) continue;
          const auto& edge = m_staticSensorLinks[static_cast<size_t>(current) * m_n + next];
          if (edge.distanceM <= kSensorRange && edge.snrDb >= kLinkSnrThresholdDb &&
              Distance(m_sensors[next].p, m_sensors[destination].p) + kEps < currentDistance) {
            candidates += 1.;
          }
        }
      }
      out.meanFeasibleRelayCandidates = candidates / out.hops;
    }
    return out;
  }
  Edge SensorAuvEdge(uint32_t sensor, uint32_t auv, double time) const {
    Edge best;
    const double d = Distance(m_sensors.at(sensor).p, AuvPosition(auv,time));
    best.distanceM = d;
    if (d > kAuvControlRange) return best;
    const double snr = SnrDb(d,kOperatingFrequencyKhz);
    if (snr >= kLinkSnrThresholdDb)
      best = {true,d,snr,AcousticRateBps(d)};
    return best;
  }
  Edge AuvMcEdge(uint32_t auv, double time) const {
    Edge best;
    const Pos mc{kControlReferenceX, kControlReferenceY};
    const double d = Distance(AuvPosition(auv, time), mc);
    best.distanceM = d;
    // This is an analytical control-plane link only.  The same 300-m AUV
    // control-range and acoustic eligibility used for sensor--AUV access are
    // applied; it can never become a payload edge.
    if (d > kAuvControlRange) return best;
    const double snr = SnrDb(d, kOperatingFrequencyKhz);
    if (snr >= kLinkSnrThresholdDb)
      best = {true, d, snr, AcousticRateBps(d)};
    return best;
  }
  Edge AuvAuvEdge(uint32_t from, uint32_t to, double time) const {
    Edge best;
    const double d = Distance(AuvPosition(from, time), AuvPosition(to, time));
    best.distanceM = d;
    if (d > kAuvControlRange) return best;
    const double snr = SnrDb(d, kOperatingFrequencyKhz);
    if (snr >= kLinkSnrThresholdDb)
      best = {true, d, snr, AcousticRateBps(d)};
    return best;
  }
  static double SerialMs(uint32_t bytes, const Edge& e) { return 1000. * bytes * 8. / e.rateBps; }
  static double EdgeDelayMs(uint32_t bytes, const Edge& e) {
    return SerialMs(bytes,e) + 1000. * PropagationSeconds(e.distanceM);
  }

  // Exact paper Nhat interpretation.  A non-progressing neighbour cannot be
  // used to estimate a finite controller-directed TD route.
  double EstimateHopsToControlReference(uint32_t i, uint32_t j) const {
    const Pos& a = m_sensors.at(i).p; const Pos& b = m_sensors.at(j).p;
    const Pos reference{kControlReferenceX,kControlReferenceY}; const double dMc = Distance(a, reference);
    if (dMc <= kEps) return 1.;
    const double projection = ((b.x-a.x)*(reference.x-a.x) + (b.y-a.y)*(reference.y-a.y)) / dMc;
    if (projection <= kEps) return std::numeric_limits<double>::infinity();
    return std::max(dMc / projection, 1.);
  }
  double TdCost(uint32_t i, uint32_t j, const Edge& e) const {
    const double nHat = EstimateHopsToControlReference(i,j);
    if (!std::isfinite(nHat)) return std::numeric_limits<double>::infinity();
    return (kPacketBits / e.rateBps + PropagationSeconds(e.distanceM)) * nHat;
  }
  double LocalEncSeconds(uint32_t node, const View& view) const {
    uint32_t received = 0;
    for (uint32_t j=0;j<m_n;++j)
      if (j!=node && view.edges.count({node,j})) ++received;
    // Paper defines ENC as hello messages received / t.  Its dimensions are
    // underspecified when added to seconds; retain the published expression as
    // a numerical time penalty rather than inventing an optimisation weight.
    return static_cast<double>(received) / HelloPeriod();
  }
  double LdpCost(uint32_t i, uint32_t /*j*/, const Edge& e, const View& view) const {
    // Paper path-duration/LDP route score: serialization + propagation +
    // connectivity (ENC) penalty.  Dijkstra therefore returns the minimum
    // valid sensor-only path duration available to this controller's view.
    return kPacketBits / e.rateBps + PropagationSeconds(e.distanceM) + LocalEncSeconds(i, view);
  }

  View LocalViewForAuv(uint32_t a, double time) const {
    View view;
    view.updatedAt=time;
    std::set<uint32_t> gateways;
    for (uint32_t i=0;i<m_n;++i) if (SensorAuvEdge(i,a,time).usable) gateways.insert(i);
    view.knownSensors=gateways;
    for (uint32_t gateway : gateways) {
      for (uint32_t j=0;j<m_n;++j) {
        if (gateway==j) continue;
        const Edge e=SensorEdge(gateway,j,time);
        if (!e.usable) continue;
        view.knownSensors.insert(j);
        view.edges[{gateway,j}]=e;
        view.edges[{j,gateway}]=SensorEdge(j,gateway,time);
      }
    }
    return view;
  }

  void Beacon() {
    const double now = Simulator::Now().GetSeconds();
    const uint64_t stateSignature = SensorStateSignature();
    const bool sensorStateChanged = !m_hasSensorStateSignature || stateSignature != m_lastSensorStateSignature;
    // At 75/100 nodes, the initial full discovery establishes the sensor
    // topology.  Subsequent updates are delta-only: with static sensor links,
    // no duplicate full HELLO/report is transmitted when the signature has not
    // changed.  This does not alter payload routing and never makes AUVs data
    // relays.  At 25/50 nodes, every scheduled sensor beacon remains counted.
    if (!gUseAggregatedTopologyDigestAccounting && (!UsesDenseSummaryUpdates() || sensorStateChanged)) {
      m_metrics.helloTx += m_n; // sensor beacons only; AUVs do not emit them
    }
    if (sensorStateChanged) {
      // MC receives aggregate gateway/sensor state only when discovery or a
      // topology/link-quality delta occurs.  Sensor topology is static here,
      // so retaining the initial view is valid between delta updates.
      View mc;
      mc.updatedAt = now;
      for (uint32_t i=0;i<m_n;++i) {
        mc.knownSensors.insert(i);
        for (uint32_t j=0;j<m_n;++j) {
          if (i==j) continue;
          const Edge e = SensorEdge(i,j,now);
          if (e.usable) mc.edges[{i,j}] = e;
        }
      }
      m_mcView = std::move(mc);
      m_lastSensorStateSignature = stateSignature;
      m_hasSensorStateSignature = true;
    }
    for (uint32_t a=0;a<kNumAuv;++a) {
      // Each AUV maintains only its own localized gateway-neighbour view; the
      // separate MC view above is the sole global topology view.
      m_views[a] = LocalViewForAuv(a,now);
    }
    if (gUseReferenceUpdateAccounting) {
      // Reference-style periodic topology accounting.  Every usable directed
      // sensor edge receives one neighbour-state update at each fixed beacon
      // epoch (the declared reception-equivalent convention).  Each LC with
      // a nonempty local view emits one aggregated local-state summary; an
      // LC--MC summary is counted only when its analytical control link is
      // currently reachable.  These are control-plane messages only.
      for (uint32_t i=0;i<m_n;++i)
        for (uint32_t j=0;j<m_n;++j)
          if (i!=j && SensorEdge(i,j,now).usable) ++m_metrics.neighborUpdateTx;
      if (gUseAggregatedTopologyDigestAccounting) {
        // Partition sensors among the three nonselected AUV LCs by nearest
        // horizontal service region.  Each LC/gateway emits ceil(region/Cap)
        // fixed-capacity topology digests.  The digest is the one aggregated
        // LC-to-controller update; it is not a payload transmission.
        std::array<uint32_t,kNumAuv> regionalSensors{};
        for (uint32_t sensor=0;sensor<m_n;++sensor) {
          uint32_t bestLc=kNumAuv;
          double bestDistance=std::numeric_limits<double>::infinity();
          for (uint32_t a=0;a<kNumAuv;++a) {
            if (a==m_selectedOc) continue;
            const double distance=Distance(m_sensors[sensor].p,AuvPosition(a,now));
            if (distance<bestDistance-1e-12 || (std::abs(distance-bestDistance)<=1e-12 && a<bestLc)) {
              bestDistance=distance; bestLc=a;
            }
          }
          if (bestLc<kNumAuv) ++regionalSensors[bestLc];
        }
        for (uint32_t a=0;a<kNumAuv;++a) {
          if (a==m_selectedOc || regionalSensors[a]==0) continue;
          m_metrics.topologyDigestTx += (regionalSensors[a]+gTopologyDigestCapacityNodes-1)/gTopologyDigestCapacityNodes;
          // One analytical acknowledgement for the LC's topology-digest
          // update epoch, independently of how many fixed-capacity digest
          // packets that LC needed in this epoch.
          ++m_metrics.mcTopologyDigestAckTx;
        }
      } else {
        for (uint32_t a=0;a<kNumAuv;++a) {
          if (!m_views[a].knownSensors.empty()) ++m_metrics.lcTopologyUpdateTx;
          if (!m_views[a].knownSensors.empty() && AuvMcEdge(a,now).usable)
            ++m_metrics.mcTopologyUpdateTx;
        }
      }
    }
    const double nextHello = HelloPeriod();
    if (now + nextHello <= kSimulationTime) Simulator::Schedule(Seconds(nextHello), &Scenario::Beacon, this);
  }

  Route ShortestRoute(uint32_t source, uint32_t destination, double time,
                      const View& view, const std::set<uint32_t>& forbidden) const {
    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> dist(m_n,inf), quality(m_n,-inf);
    std::vector<uint32_t> hops(m_n,std::numeric_limits<uint32_t>::max());
    std::vector<int32_t> prev(m_n,-1);
    using Item=std::pair<double,uint32_t>;
    std::priority_queue<Item,std::vector<Item>,std::greater<Item>> q;
    if (!view.knownSensors.count(source) || !view.knownSensors.count(destination) || forbidden.count(source)) return {};
    dist[source]=0.; hops[source]=0; quality[source]=0.; q.push({0.,source});
    while(!q.empty()) {
      const auto [cost,u]=q.top();q.pop(); if (cost!=dist[u]) continue;
      for(uint32_t v=0;v<m_n;++v) {
        if(u==v || forbidden.count(v) || !view.knownSensors.count(v)) continue;
        const auto it=view.edges.find({u,v});
        if(it==view.edges.end()) continue;
        const Edge e=it->second;
        if(!e.usable) continue;
        const double w=LdpCost(u,v,e,view);
        const double candidateCost=cost+w;
        const uint32_t candidateHops=hops[u]+1;
        const double candidateQuality=quality[u]+e.snrDb;
        // Paper-style path duration is primary.  Exact/equivalent path-cost
        // ties are deterministic: fewer sensor hops, then higher cumulative
        // analytical link SNR, then the existing node-ID queue ordering.
        const bool betterCost=candidateCost < dist[v]-1e-12;
        const bool tiedCost=std::abs(candidateCost-dist[v])<=1e-12;
        const bool betterTie=tiedCost &&
          (candidateHops<hops[v] ||
           (candidateHops==hops[v] && candidateQuality>quality[v]+1e-12));
        if(betterCost || betterTie) {
          dist[v]=candidateCost; hops[v]=candidateHops; quality[v]=candidateQuality;
          prev[v]=static_cast<int32_t>(u);q.push({dist[v],v});
        }
      }
    }
    if(!std::isfinite(dist[destination])) return {};
    Route route;route.valid=true;route.cost=dist[destination];
    for(int32_t at=static_cast<int32_t>(destination);at>=0;at=prev[at]) {
      route.nodes.push_back(static_cast<uint32_t>(at)); if(static_cast<uint32_t>(at)==source) break;
    }
    if(route.nodes.empty() || route.nodes.back()!=source) return {};
    std::reverse(route.nodes.begin(),route.nodes.end());
    uint64_t h=1469598103934665603ULL;
    for(uint32_t n:route.nodes) h=FnvMix(h,n);
    route.hash=h; return route;
  }

  // Sensor-to-gateway TD path, used only for control access when direct OC
  // contact is unavailable.  It follows equations (1)-(2), never carries data.
  bool ControlAccessToAuv(uint32_t source, uint32_t auv, double time,
                          double& oneWayMs, uint32_t& hops, bool accountGateway) {
    const Edge direct=SensorAuvEdge(source,auv,time);
    if(direct.usable) { oneWayMs=EdgeDelayMs(kControlBytes,direct);hops=1;return true; }
    std::set<uint32_t> gateways;
    for(uint32_t s=0;s<m_n;++s) if(SensorAuvEdge(s,auv,time).usable) gateways.insert(s);
    if(gateways.empty()) return false;
    const double inf=std::numeric_limits<double>::infinity();
    std::vector<double>d(m_n,inf); std::vector<uint32_t> h(m_n,0);
    using Item=std::pair<double,uint32_t>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>>q;
    d[source]=0.;q.push({0.,source});uint32_t target=m_n;
    while(!q.empty()) { auto [cost,u]=q.top();q.pop();if(cost!=d[u])continue;if(gateways.count(u)){target=u;break;}
      for(uint32_t v=0;v<m_n;++v){if(u==v)continue;const Edge e=SensorEdge(u,v,time);if(!e.usable)continue;
        const double w=TdCost(u,v,e);if(!std::isfinite(w))continue;
        if(cost+w<d[v]){d[v]=cost+w;h[v]=h[u]+1;q.push({d[v],v});}}
    }
    if(target==m_n)return false;
    const Edge up=SensorAuvEdge(target,auv,time);if(!up.usable)return false;
    oneWayMs=1000.*d[target]+EdgeDelayMs(kControlBytes,up);hops=h[target]+1;
    // This is a diagnostic subset of route-request/reply transport, not an
    // additional control class.  Count its two directions here but never add
    // it again to total ROR control transmissions.
    if (accountGateway) m_metrics.gatewayControlTx += 2*h[target];
    return true;
  }

  bool SelectLcAccess(uint32_t source, double time, uint32_t& lc,
                      double& oneWayMs, uint32_t& hops) {
    bool found = false;
    double best = std::numeric_limits<double>::infinity();
    // The other three AUVs are LCs.  Prefer the reachable LC with the least
    // control-plane delay; tie-breaking by AUV ID is deterministic.
    for (uint32_t candidate=0; candidate<kNumAuv; ++candidate) {
      if (candidate == m_selectedOc) continue;
      double candidateDelay=0.; uint32_t candidateHops=0;
      if (!ControlAccessToAuv(source, candidate, time, candidateDelay, candidateHops, false)) continue;
      if (!found || candidateDelay < best - 1e-12 ||
          (std::abs(candidateDelay-best) <= 1e-12 && candidate < lc)) {
        found=true; best=candidateDelay; lc=candidate; oneWayMs=candidateDelay; hops=candidateHops;
      }
    }
    // If no nonselected LC can be reached, direct/gateway access to the OC is
    // the paper's permitted controller-access fallback, not a data relay.
    if (!found) {
      lc=m_selectedOc;
      if (!ControlAccessToAuv(source, lc, time, oneWayMs, hops, false)) return false;
    }
    // Repeat the chosen deterministic access calculation only to account for
    // realized gateway control transmissions.
    return ControlAccessToAuv(source, lc, time, oneWayMs, hops, true);
  }

  bool OcMcControlAccess(double time, double& oneWayMs) const {
    const Edge edge = AuvMcEdge(m_selectedOc, time);
    if (!edge.usable) return false;
    oneWayMs = EdgeDelayMs(kControlBytes, edge);
    return true;
  }

  bool InstallRoute(uint32_t flowId, uint32_t current, const std::set<uint32_t>& forbidden,
                    bool recovery, bool& mcFallbackAttempted, double& controlWaitMs) {
    const double now=Simulator::Now().GetSeconds();
    const uint32_t destination=m_flows[flowId].destination;
    const std::pair<uint32_t,uint32_t> key{current,destination};
    // A no-path entry is checked before generating any new reactive-control
    // traffic.  Expiry is short and independent of both HELLO and route TTL.
    auto negative=m_negativeRouteCache.find(key);
    if (negative!=m_negativeRouteCache.end()) {
      if (negative->second.expiresAt>=now) {
        ++m_metrics.negativeCacheHits;
        // A negative-cache hit is a suppressed packet-level outcome, not a
        // fresh route-discovery attempt and therefore creates no new request
        // or failure-control transaction.
        return false;
      }
      m_negativeRouteCache.erase(negative);
    }
    // Current control resolution is synchronous in this analytical model, so
    // normal packet scheduling does not overlap requests.  The guard still
    // prevents duplicate requests if that invariant ever changes.
    if (!m_pendingRouteRequests.insert(key).second) {
      ++m_metrics.pendingRouteSuppressions;
      ++m_metrics.failedRouteAttempts;
      return false;
    }
    const auto fail=[&]() {
      ++m_metrics.failedRouteAttempts;
      m_pendingRouteRequests.erase(key);
      return false;
    };
    double oneWay=0.;uint32_t controlHops=0, lc=m_selectedOc;
    ++m_metrics.controllerRequests;
    if(!SelectLcAccess(current,now,lc,oneWay,controlHops)) { ++m_metrics.controlUnreachable; return fail(); }
    m_metrics.routeRequestTx += controlHops; m_metrics.routeReplyTx += controlHops;
    controlWaitMs += 2.*oneWay + kControllerLookupMs; // requester <-> LC

    // A valid LC cache entry answers directly, avoiding OC/MC transport.
    if (!recovery) {
      auto lcIt=m_lcRouteCaches[lc].find(key);
      if (lcIt!=m_lcRouteCaches[lc].end() && lcIt->second.expiresAt >= now &&
          RouteValid(lcIt->second.route,now)) {
        ++m_metrics.lcCacheHits;
        auto& state=m_flowStates[flowId];
        state.installed=lcIt->second.route; state.expiresAt=lcIt->second.expiresAt;
        ++state.setupSuccesses; m_lastRouteHash=state.installed.hash;
        m_pendingRouteRequests.erase(key);
        return true;
      }
    }
    ++m_metrics.lcCacheMisses;

    // Forward an LC miss to the ML-selected OC.  When the OC itself was the
    // only reachable controller, this hop is zero because no LC-to-OC relay
    // exists; neither case carries payload.
    if (lc != m_selectedOc) {
      const Edge lcOc=AuvAuvEdge(lc,m_selectedOc,now);
      if (!lcOc.usable) { ++m_metrics.controlUnreachable; return fail(); }
      const double lcOcOneWay=EdgeDelayMs(kControlBytes,lcOc);
      m_metrics.routeRequestTx += 1; m_metrics.routeReplyTx += 1;
      controlWaitMs += 2.*lcOcOneWay + kControllerLookupMs;
    }

    Route route;
    if (!recovery) {
      auto ocIt=m_ocRouteCache.find(key);
      if (ocIt!=m_ocRouteCache.end() && ocIt->second.expiresAt >= now &&
          RouteValid(ocIt->second.route,now)) {
        ++m_metrics.ocCacheHits;
        route=ocIt->second.route;
      }
    }
    if (!route.valid) {
      ++m_metrics.ocCacheMisses; ++m_metrics.ocRouteComputations;
      const View& view=m_views[m_selectedOc];
      route=ShortestRoute(current,destination,now,view,forbidden);
      if(route.valid) ++m_metrics.ocRouteSuccesses;
    }
    if (!route.valid) {
      ++m_metrics.ocRouteFailures;

      // Partner-style/simple underwater run: do not let the MC rescue every bad
      // OC decision.  The selected OC must resolve the route from its own local
      // view.  This is what makes selected-OC quality affect PDR as well as
      // delay/ROR.  The MC remains in the architecture as the surface controller
      // reference, but it is not used as a global route fallback in this mode.
      if (gDisableMcFallback) {
        m_negativeRouteCache[key]={now+NegativeRouteTtl()};
        return fail();
      }

      // Full hierarchical mode: LC/OC miss asks the surface MC.  The MC's
      // aggregate view is used only to return a sensor-only path; neither OC
      // nor MC can become a payload next hop.
      if (mcFallbackAttempted) return fail();
      mcFallbackAttempted=true;
      ++m_metrics.mcFallbacks;
      double mcOneWay=0.;
      if (!OcMcControlAccess(now, mcOneWay)) {
        ++m_metrics.controlUnreachable;
        return fail();
      }
      m_metrics.mcControlTx += 2; // analytical OC->MC request and MC->OC reply
      controlWaitMs += 2.*mcOneWay + kControllerLookupMs;
      ++m_metrics.mcRouteComputations;
      route=ShortestRoute(current,destination,now,m_mcView,forbidden);
      if (route.valid) ++m_metrics.mcRouteSuccesses;
      else {
        ++m_metrics.mcRouteFailures;
        m_negativeRouteCache[key]={now+NegativeRouteTtl()};
        return fail();
      }
    }
    auto& state=m_flowStates[flowId];
    state.installed=route; state.expiresAt=now+RouteTtl();
    // The route is returned through the selected OC to the querying LC.
    // MC and OC success responses are installed in the selected OC and
    // reaching LC caches.  A recovery is keyed by its current sensor and
    // destination, so it cannot be mistaken for a source-originating route.
    const CachedRoute entry{route, state.expiresAt};
    m_ocRouteCache[key]=entry;
    m_lcRouteCaches[lc][key]=entry;
    m_negativeRouteCache.erase(key);
    ++state.setupSuccesses; m_lastRouteHash=route.hash;
    m_pendingRouteRequests.erase(key);
    return true;
  }

  bool RouteValid(const Route& route,double time) const {
    if(!route.valid||route.nodes.size()<2)return false;
    for(size_t i=0;i+1<route.nodes.size();++i) if(!SensorEdge(route.nodes[i],route.nodes[i+1],time).usable)return false;
    return true;
  }

  void LaunchPacket(uint32_t flowId, uint32_t packetOrdinal) {
    auto& fs=m_flowStates[flowId]; ++fs.generated; ++m_metrics.generated;
    AttemptRouteThenLaunch(flowId, packetOrdinal, Simulator::Now(), 0, 0.);
  }

  void AttemptRouteThenLaunch(uint32_t flowId, uint32_t packetOrdinal,
                              Time generatedAt, uint32_t retryOrdinal,
                              double accumulatedControlWaitMs) {
    auto& fs=m_flowStates[flowId];
    const double now=Simulator::Now().GetSeconds();
    double controlWait=0.;
    bool mustInstall=!RouteValid(fs.installed,now) || fs.expiresAt<now ||
                     fs.installed.nodes.empty() || fs.installed.nodes.front()!=m_flows[flowId].source;
    bool mcFallbackAttempted=false;
    if(mustInstall) {
      ++fs.setupAttempts; if(fs.installed.valid) ++fs.reestablishments;
      if(!InstallRoute(flowId,m_flows[flowId].source,{},false,mcFallbackAttempted,controlWait)) {
        const double elapsed=(Simulator::Now()-generatedAt).GetSeconds();
        const double retryDelay=RouteRetryBackoffSeconds(retryOrdinal);
        const double retryWait=controlWait / 1000. + retryDelay;
        if (gBufferRouteDiscovery && retryOrdinal < gRouteDiscoveryRetryLimit &&
            elapsed + retryWait <= gPacketBufferTimeoutSeconds &&
            now + retryWait <= kSimulationTime) {
          ++m_metrics.routeDiscoveryBufferedRetries;
          Simulator::Schedule(Seconds(retryWait), &Scenario::AttemptRouteThenLaunch, this,
                              flowId, packetOrdinal, generatedAt, retryOrdinal+1,
                              accumulatedControlWaitMs+controlWait);
          return;
        }
        ++fs.setupFailures; ++fs.droppedNoRoute; ++m_metrics.finalNoPath;
        if (gBufferRouteDiscovery && retryOrdinal > 0) ++m_metrics.routeDiscoveryTimeoutDrops;
        if(!PhysicalPathExists(m_flows[flowId].source,m_flows[flowId].destination,now)) ++m_metrics.noPhysicalPath;
        else ++m_metrics.routingFailureWithPath;
        return;
      }
    } else ++m_metrics.installedRouteHits;
    PacketState st{flowId,packetOrdinal,fs.installed,0,generatedAt,
                   accumulatedControlWaitMs+controlWait,mcFallbackAttempted,{m_flows[flowId].source},0.,0,0.,0.};
    Simulator::Schedule(MilliSeconds(controlWait),&Scenario::ForwardHop,this,st);
  }

  struct PacketState {
    uint32_t flowId, packetOrdinal; Route route; size_t at; Time generatedAt; double controllerWaitMs;
    bool mcFallbackAttempted;
    std::set<uint32_t> visited;
    double traversedLengthM;
    uint32_t traversedHops;
    double relayCandidateCountTotal;
    double relayWaitMsTotal;
  };
  uint32_t PositiveProgressCandidates(uint32_t from, uint32_t destination, double time) const {
    const double currentDistance=Distance(m_sensors[from].p,m_sensors[destination].p);
    uint32_t count=0;
    for(uint32_t candidate=0;candidate<m_n;++candidate) {
      if(candidate==from) continue;
      const Edge edge=SensorEdge(from,candidate,time);
      if(edge.usable && Distance(m_sensors[candidate].p,m_sensors[destination].p) < currentDistance-1e-9)
        ++count;
    }
    return count;
  }
  void ForwardHop(PacketState st) {
    if(st.at+1>=st.route.nodes.size()) { Deliver(st);return; }
    const uint32_t from=st.route.nodes[st.at], to=st.route.nodes[st.at+1];
    if(from>=m_n||to>=m_n) { ++m_metrics.architectureViolations; NS_ABORT_MSG("DATA packet attempted AUV/OC hop"); }
    Edge actual=SensorEdge(from,to,Simulator::Now().GetSeconds());
    if(!actual.usable) {
      ++m_metrics.linkInvalidations;
      const double distance=Distance(m_sensors[from].p,m_sensors[to].p);
      if(distance>kSensorRange) ++m_metrics.distanceFailures;
      else ++m_metrics.snrFailures;
      Recover(st,from); return;
    }
    const double serial=SerialMs(kPacketBytes,actual), propagation=1000.*PropagationSeconds(actual.distanceM);
    const uint32_t candidateCount=PositiveProgressCandidates(from,st.route.nodes.back(),Simulator::Now().GetSeconds());
    double relayWait=0.;
    if(gUseRelaySearchDelay) {
      const double candidates=std::max(1.,static_cast<double>(candidateCount));
      relayWait=std::min(gRelaySearchBaseMs + gRelaySearchScaleMs/std::pow(candidates,gRelaySearchCandidateExponent),
                         gRelaySearchMaxMs);
    }
    ++m_metrics.dataHops; m_metrics.serializationMsTotal+=serial; m_metrics.propagationMsTotal+=propagation;
    st.traversedLengthM += actual.distanceM;
    ++st.traversedHops;
    st.relayCandidateCountTotal += candidateCount;
    st.relayWaitMsTotal += relayWait;
    if(m_trace) std::cout<<"TRACE flow="<<st.flowId<<" packet="<<st.packetOrdinal<<" "<<from<<"->"<<to
                         <<" f_kHz="<<kOperatingFrequencyKhz<<" snr="<<actual.snrDb<<" route="<<Hex(st.route.hash)<<"\n";
    st.visited.insert(from);++st.at;
    Simulator::Schedule(MilliSeconds(relayWait+serial+propagation),&Scenario::ForwardHop,this,st);
  }
  void Recover(PacketState st,uint32_t current) {
    ++m_metrics.routeRecoveries;
    double wait=0.;
    if(!InstallRoute(st.flowId,current,st.visited,true,st.mcFallbackAttempted,wait)) {
      ++m_metrics.recoveryFailures; ++m_metrics.finalNoPath;
      const double now=Simulator::Now().GetSeconds();
      if(!PhysicalPathExists(current,m_flows[st.flowId].destination,now)) ++m_metrics.noPhysicalPath;
      else ++m_metrics.routingFailureWithPath;
      return;
    }
    ++m_metrics.recoverySuccesses;
    st.route=m_flowStates[st.flowId].installed;st.at=0;st.controllerWaitMs+=wait;st.visited={current};
    Simulator::Schedule(MilliSeconds(wait),&Scenario::ForwardHop,this,st);
  }
  void Deliver(const PacketState& st) {
    ++m_metrics.delivered; ++m_flowStates[st.flowId].delivered;
    const double total=(Simulator::Now()-st.generatedAt).GetMilliSeconds();
    m_metrics.delayMsTotal+=total; m_metrics.controllerWaitMsTotal+=st.controllerWaitMs;
    m_metrics.deliveredPathLengthMTotal += st.traversedLengthM;
    m_metrics.deliveredHopCountTotal += st.traversedHops;
    m_metrics.deliveredRelayCandidateCountTotal += st.relayCandidateCountTotal;
    m_metrics.deliveredRelayWaitMsTotal += st.relayWaitMsTotal;
  }
  bool PhysicalPathExists(uint32_t source,uint32_t destination,double time) const {
    std::vector<bool> seen(m_n);std::queue<uint32_t>q;q.push(source);seen[source]=true;
    while(!q.empty()){const auto u=q.front();q.pop();if(u==destination)return true;
      for(uint32_t v=0;v<m_n;++v)if(!seen[v]&&u!=v&&SensorEdge(u,v,time).usable){seen[v]=true;q.push(v);}}
    return false;
  }
};

static uint32_t SelectOcLabel(const std::vector<Auv>& auvs, const std::vector<double>& density) {
  std::array<double,4> center{}, d{}, c{}, s{}, score{};
  double dMin=std::numeric_limits<double>::infinity(),dMax=0.,nMin=std::numeric_limits<double>::infinity(),nMax=0.,vMin=std::numeric_limits<double>::infinity(),vMax=0.;
  for(uint32_t i=0;i<4;++i){center[i]=Distance(auvs[i].start,{kControlReferenceX,kControlReferenceY});dMin=std::min(dMin,center[i]);dMax=std::max(dMax,center[i]);nMin=std::min(nMin,density[i]);nMax=std::max(nMax,density[i]);vMin=std::min(vMin,auvs[i].speed);vMax=std::max(vMax,auvs[i].speed);}
  for(uint32_t i=0;i<4;++i){d[i]=(nMax==nMin)?1.:(density[i]-nMin)/(nMax-nMin);c[i]=(dMax==dMin)?1.:1.-(center[i]-dMin)/(dMax-dMin);s[i]=(vMax==vMin)?1.:1.-(auvs[i].speed-vMin)/(vMax-vMin);score[i]=(d[i]+c[i]+s[i])/3.;}
  uint32_t best=0;for(uint32_t i=1;i<4;++i){if(score[i]>score[best]+1e-12 || (std::abs(score[i]-score[best])<=1e-12 && (density[i]>density[best] || (density[i]==density[best] && (center[i]<center[best] || (center[i]==center[best] && (auvs[i].speed<auvs[best].speed || (auvs[i].speed==auvs[best].speed && i<best))))))))best=i;}return best;
}

static void WriteLabelHeader(std::ostream& out) { out<<"scenario_id,scenario_seed,node_count,auv_id,x,y,local_density,speed,is_oc\n"; }
static void WriteLabelRows(std::ostream& out, uint32_t scenarioId, uint32_t seed, uint32_t n) {
  Scenario s(seed,n,scenarioId,0,false); const auto density=s.LocalDensities();const auto& auvs=s.Auvs();const uint32_t label=SelectOcLabel(auvs,density);
  for(uint32_t a=0;a<4;++a)out<<scenarioId<<','<<seed<<','<<n<<','<<a<<','<<std::setprecision(12)<<auvs[a].start.x<<','<<auvs[a].start.y<<','<<density[a]<<','<<auvs[a].speed<<','<<(a==label?1:0)<<'\n';
  Simulator::Destroy();
}

// Pre-routing candidate features.  They are deterministic functions of the
// deployment, scheduled flows and initial AUV state, exported before a route
// is installed or a payload packet is generated.  Source/destination values
// are means (or coverage fractions) across the ten scheduled flows.
static void WriteTopologyFeatureHeader(std::ostream& out) {
  out << "scenario_id,scenario_seed,node_count,auv_id,x,y,local_density,speed,"
      << "source_to_oc_distance,destination_to_oc_distance,sensors_in_oc_range,"
      << "gateways_in_oc_range,source_covered_by_oc,destination_covered_by_oc,"
      << "estimated_local_path_exists,average_link_quality_in_oc_view,"
      << "estimated_hop_count,mean_source_destination_distance_m,"
      << "connected_flow_pair_ratio,mean_static_shortest_hops,"
      << "mean_static_shortest_path_length_m,mean_static_feasible_relay_candidates,"
      << "topology_hash\n";
}

static void WriteTopologyFeatureRows(std::ostream& out, uint32_t scenarioId,
                                     uint32_t seed, uint32_t n) {
  Scenario s(seed, n, scenarioId, 0, false);
  const auto& auvs = s.Auvs();
  const auto density = s.LocalDensities();
  const auto pre=s.PreRoutingOcFeatures();
  for (uint32_t a = 0; a < kNumAuv; ++a) {
    const Pos auv = auvs[a].start;
    const auto& f=pre[a];
    out << scenarioId << ',' << seed << ',' << n << ',' << a << ','
        << std::setprecision(12) << auv.x << ',' << auv.y << ',' << density[a] << ','
        << auvs[a].speed << ',' << f.sourceDistanceM << ',' << f.destinationDistanceM << ','
        << f.sensorsInRange << ',' << f.gatewaysInRange << ',' << f.sourceCoveredFraction << ','
        << f.destinationCoveredFraction << ',' << f.localPathCoverage << ','
        << f.averageLocalLinkQuality << ',' << f.estimatedHopCount << ','
        << s.MeanSourceDestinationDistance() << ',' << s.ConnectedFlowPairRatio() << ','
        << s.MeanStaticShortestHops() << ',' << s.MeanStaticShortestPathLengthM() << ','
        << s.MeanStaticFeasibleRelayCandidates() << ','
        << Hex(s.TopologyHash()) << '\n';
  }
  Simulator::Destroy();
}

struct NetworkLabelMetric {
  double pdr = 0., e2ed = std::numeric_limits<double>::quiet_NaN(), ror = 0.;
  uint32_t tx = 0, rx = 0;
};

static NetworkLabelMetric RunForNetworkLabel(uint32_t seed, uint32_t n, uint32_t scenarioId, uint32_t oc) {
  Scenario trial(seed,n,scenarioId,oc,false);
  trial.Run();
  const Metrics& m=trial.GetMetrics();
  const double pdr=m.generated ? 100.*m.delivered/m.generated : 0.;
  const double e2ed=m.delivered ? m.delayMsTotal/m.delivered : std::numeric_limits<double>::quiet_NaN();
  // Underwater-reference-paper ROR: control-packet transmissions divided by
  // all realized control and payload-hop transmissions in the network.
  const double control=TotalControlPackets(m);
  const double ror=(control+m.dataHops) ? control/(control+m.dataHops) : 0.;
  const NetworkLabelMetric result{pdr,e2ed,ror,m.generated,m.delivered};
  Simulator::Destroy();
  return result;
}

static bool BetterNetworkLabel(const NetworkLabelMetric& candidate, uint32_t candidateId,
                               const NetworkLabelMetric& incumbent, uint32_t incumbentId) {
  constexpr double eps=1e-9;
  if(candidate.pdr > incumbent.pdr + eps) return true;
  if(incumbent.pdr > candidate.pdr + eps) return false;
  // PDR=0 candidates have undefined E2ED.  A defined delay is preferred only
  // when a delivery exists; otherwise ROR resolves the equal-PDR comparison.
  const bool cDefined=std::isfinite(candidate.e2ed), iDefined=std::isfinite(incumbent.e2ed);
  if(cDefined != iDefined) return cDefined;
  if(cDefined && std::abs(candidate.e2ed-incumbent.e2ed)>eps)
    return candidate.e2ed < incumbent.e2ed;
  if(std::abs(candidate.ror-incumbent.ror)>eps) return candidate.ror < incumbent.ror;
  return candidateId < incumbentId;
}

// Performance-labelled dataset utility.  Network outcomes appear only here to
// select one label per scenario; they are written to a separate diagnostic
// ledger and never written as ML input columns.
static void WriteNetworkLabelRows(std::ostream& out, std::ostream& diagnostic,
                                  uint32_t scenarioId, uint32_t seed, uint32_t n) {
  std::vector<Auv> auvs; std::vector<double> density;
  {
    Scenario features(seed,n,scenarioId,0,false);
    auvs=features.Auvs(); density=features.LocalDensities();
    Simulator::Destroy();
  }
  std::array<NetworkLabelMetric,4> metrics;
  uint32_t best=0;
  for(uint32_t oc=0;oc<4;++oc) {
    metrics[oc]=RunForNetworkLabel(seed,n,scenarioId,oc);
    if(oc && BetterNetworkLabel(metrics[oc],oc,metrics[best],best)) best=oc;
  }
  for(uint32_t a=0;a<4;++a) {
    out<<scenarioId<<','<<seed<<','<<n<<','<<a<<','<<std::setprecision(12)
       <<auvs[a].start.x<<','<<auvs[a].start.y<<','<<density[a]<<','<<auvs[a].speed
       <<','<<(a==best?1:0)<<'\n';
    diagnostic<<scenarioId<<','<<seed<<','<<n<<','<<a<<','<<metrics[a].tx<<','<<metrics[a].rx<<','
              <<std::fixed<<std::setprecision(6)<<metrics[a].pdr<<',';
    if(std::isfinite(metrics[a].e2ed)) diagnostic<<metrics[a].e2ed; else diagnostic<<"NA";
    diagnostic<<','<<metrics[a].ror<<','<<(a==best?1:0)<<'\n';
  }
}

static void WriteRunHeader(std::ostream& out) {
  out<<"scenario_id,scenario_seed,node_count,selected_oc,x,y,local_density,speed,topology_hash,acoustic_state_hash,route_hash,generated_packets,delivered_packets,PDR,E2ED_ms,mean_source_destination_distance_m,connected_flow_pair_ratio,mean_static_shortest_hops,mean_static_shortest_path_length_m,mean_static_feasible_relay_candidates,mean_delivered_path_length_m,mean_delivered_hop_count,mean_delivered_relay_candidate_count,mean_delivered_relay_wait_ms,ROR_total,ROR_generated,ROR_hello,ROR_reactive,hello_tx,neighbor_update_tx,raw_neighbor_update_burden,compressed_topology_updates,topology_digest_packets,mc_topology_digest_ack_packets,lc_topology_update_tx,mc_topology_update_tx,route_request_tx,route_reply_tx,mc_control_tx,gateway_control_tx,control_transmissions,data_hops,OC_data_hops,architecture_violations,controller_requests,installed_route_hits,lc_cache_hits,lc_cache_misses,oc_route_computations,oc_route_successes,oc_route_failures,oc_cache_hits,oc_cache_misses,mc_fallbacks,mc_route_computations,mc_route_successes,mc_route_failures,route_recoveries,recovery_successes,recovery_failures,link_invalidations,control_unreachable,no_physical_path,routing_failure_with_path,final_no_path_drops,failed_route_attempts,negative_cache_hits,pending_route_suppressions,route_discovery_buffered_retries,route_discovery_timeout_drops\n";
}
static void WriteRun(std::ostream& out,uint32_t scenarioId,uint32_t seed,uint32_t n,uint32_t oc,bool trace) {
  Scenario s(seed,n,scenarioId,oc,trace);s.Run();const auto&m=s.GetMetrics();
  const auto density=s.LocalDensities(); const auto& auvs=s.Auvs();
  const double pdr=m.generated?100.*m.delivered/m.generated:0.; const double e2e=m.delivered?m.delayMsTotal/m.delivered:std::numeric_limits<double>::quiet_NaN();
  const double meanPathLength=m.delivered?m.deliveredPathLengthMTotal/m.delivered:std::numeric_limits<double>::quiet_NaN();
  const double meanHops=m.delivered?static_cast<double>(m.deliveredHopCountTotal)/m.delivered:std::numeric_limits<double>::quiet_NaN();
  const double meanRelayCandidates=m.deliveredHopCountTotal ? m.deliveredRelayCandidateCountTotal/m.deliveredHopCountTotal : std::numeric_limits<double>::quiet_NaN();
  const double meanRelayWait=m.deliveredHopCountTotal ? m.deliveredRelayWaitMsTotal/m.deliveredHopCountTotal : std::numeric_limits<double>::quiet_NaN();
  // Underwater-reference-paper primary ROR: all realized control-packet
  // transmissions divided by control plus realized payload-hop transmissions.
  const double control=TotalControlPackets(m);
  const double compressedTopology=CompressedTopologyUpdates(m);
  const double rorTotal=(control+m.dataHops)?control/(control+m.dataHops):0.;
  const double rorGenerated=(control+m.generated)?control/(control+m.generated):0.;
  const double rorHello=(m.helloTx+m.dataHops)
      ? static_cast<double>(m.helloTx)/(m.helloTx+m.dataHops) : 0.;
  const double reactive=m.routeRequestTx+m.routeReplyTx+m.mcControlTx;
  const double rorReactive=(reactive+m.dataHops)?reactive/(reactive+m.dataHops):0.;
  out<<scenarioId<<','<<seed<<','<<n<<','<<oc<<','<<std::setprecision(12)
     <<auvs[oc].start.x<<','<<auvs[oc].start.y<<','<<density[oc]<<','<<auvs[oc].speed<<','
     <<Hex(s.TopologyHash())<<','<<Hex(s.AcousticStateHash())<<','<<Hex(s.LastRouteHash())<<','
     <<m.generated<<','<<m.delivered<<','<<std::fixed<<std::setprecision(6)<<pdr<<',';
  if(std::isfinite(e2e))out<<e2e;else out<<"NA";
  out<<','<<s.MeanSourceDestinationDistance()<<','<<s.ConnectedFlowPairRatio()<<','
     <<s.MeanStaticShortestHops()<<','<<s.MeanStaticShortestPathLengthM()<<','
     <<s.MeanStaticFeasibleRelayCandidates()<<',';
  if(std::isfinite(meanPathLength))out<<meanPathLength;else out<<"NA";
  out<<',';
  if(std::isfinite(meanHops))out<<meanHops;else out<<"NA";
  out<<',';
  if(std::isfinite(meanRelayCandidates))out<<meanRelayCandidates;else out<<"NA";
  out<<',';
  if(std::isfinite(meanRelayWait))out<<meanRelayWait;else out<<"NA";
  out<<','<<rorTotal<<','<<rorGenerated<<','<<rorHello<<','<<rorReactive<<','<<m.helloTx<<','<<m.neighborUpdateTx<<','<<m.neighborUpdateTx<<','<<compressedTopology<<','<<m.topologyDigestTx<<','<<m.mcTopologyDigestAckTx<<','<<m.lcTopologyUpdateTx<<','<<m.mcTopologyUpdateTx<<','<<m.routeRequestTx<<','<<m.routeReplyTx<<','<<m.mcControlTx<<','<<m.gatewayControlTx<<','<<control<<','<<m.dataHops<<','<<m.ocDataHops<<','<<m.architectureViolations<<','<<m.controllerRequests<<','<<m.installedRouteHits<<','<<m.lcCacheHits<<','<<m.lcCacheMisses<<','<<m.ocRouteComputations<<','<<m.ocRouteSuccesses<<','<<m.ocRouteFailures<<','<<m.ocCacheHits<<','<<m.ocCacheMisses<<','<<m.mcFallbacks<<','<<m.mcRouteComputations<<','<<m.mcRouteSuccesses<<','<<m.mcRouteFailures<<','<<m.routeRecoveries<<','<<m.recoverySuccesses<<','<<m.recoveryFailures<<','<<m.linkInvalidations<<','<<m.controlUnreachable<<','<<m.noPhysicalPath<<','<<m.routingFailureWithPath<<','<<m.finalNoPath<<','<<m.failedRouteAttempts<<','<<m.negativeCacheHits<<','<<m.pendingRouteSuppressions<<','<<m.routeDiscoveryBufferedRetries<<','<<m.routeDiscoveryTimeoutDrops<<'\n';
  std::cout<<"scenario="<<scenarioId<<" OC"<<oc<<" TX="<<m.generated<<" RX="<<m.delivered<<" PDR="<<std::setprecision(2)<<pdr<<" E2ED="<<e2e<<" ROR="<<rorTotal<<" route="<<Hex(s.LastRouteHash())<<"\n";
  Simulator::Destroy();
}

} // namespace

int main(int argc, char* argv[]) {
  uint32_t nodeCount=50, scenarioSeed=29, scenarioId=0, selectedOc=0, runs=1;
  uint32_t baseSeed=29; std::string mode="run", output="results/underwater_rebuild/runs.csv", scenarioManifest="";
  std::string diagnosticOutput=""; bool trace=false, fixedControlPolicy=false;
  double helloSeconds=kHelloPeriod, negativeRouteTtlSeconds=kNegativeRouteTtl,
         routeTtlSeconds=kRouteTtl;
  double packetBufferTimeoutSeconds=0.; uint32_t routeDiscoveryRetryLimit=0;
  bool referenceUpdateAccounting=false, bufferRouteDiscovery=false, longDistanceFlows=false, relaySearchDelay=false, disableMcFallback=false;
  bool balancedDistanceFlows=false, hopBalancedDistanceFlows=false;
  double relaySearchBaseMs=5., relaySearchScaleMs=180., relaySearchMaxMs=80., relaySearchCandidateExponent=1.;
  double topologyUpdateCompressionFactor=1.;
  bool aggregatedTopologyDigestAccounting=false; uint32_t topologyDigestCapacityNodes=25;
  CommandLine cmd(__FILE__);
  cmd.AddValue("mode","run, labels, network-labels, or topology-features",mode);
  cmd.AddValue("nodeCount","Sensor-node count",nodeCount); cmd.AddValue("scenarioSeed","Exact deterministic scenario seed",scenarioSeed);
  cmd.AddValue("scenarioId","Scenario identifier",scenarioId);cmd.AddValue("selectedOc","Selected controller candidate (0..3)",selectedOc);
  cmd.AddValue("runs","Number of consecutive scenarios",runs);cmd.AddValue("baseSeed","First seed for --runs",baseSeed);
  cmd.AddValue("scenarioManifest","Optional CSV of already accepted scenario_id,scenario_seed records",scenarioManifest);
  cmd.AddValue("output","CSV output path",output);cmd.AddValue("diagnosticOutput","Network-label diagnostic CSV path",diagnosticOutput);
  cmd.AddValue("trace","Emit forwarding trace",trace);
  cmd.AddValue("fixedControlPolicy","Use one fixed HELLO/negative-cache policy at every density",fixedControlPolicy);
  cmd.AddValue("helloSeconds","Fixed-control HELLO interval in seconds",helloSeconds);
  cmd.AddValue("negativeRouteTtlSeconds","Fixed-control no-path cache TTL in seconds",negativeRouteTtlSeconds);
  cmd.AddValue("routeTtlSeconds","Fixed-control installed-route TTL in seconds",routeTtlSeconds);
  cmd.AddValue("referenceUpdateAccounting","Account periodic neighbour/LC/MC topology updates",referenceUpdateAccounting);
  cmd.AddValue("bufferRouteDiscovery","Buffer a packet while route-control access is retried",bufferRouteDiscovery);
  cmd.AddValue("disableMcFallback","Partner-style simple mode: selected OC must resolve route locally; no MC global rescue",disableMcFallback);
  cmd.AddValue("routeDiscoveryRetryLimit","Maximum buffered route-discovery retries",routeDiscoveryRetryLimit);
  cmd.AddValue("packetBufferTimeoutSeconds","Maximum packet route-discovery buffer lifetime",packetBufferTimeoutSeconds);
  cmd.AddValue("longDistanceFlows","Choose all traffic pairs at least 424.3 m apart",longDistanceFlows);
  cmd.AddValue("balancedDistanceFlows","Choose 300--450 m flows, preferring static sensor-connected pairs",balancedDistanceFlows);
  cmd.AddValue("hopBalancedDistanceFlows","Choose only connected 300--450 m flows in the 5--8-hop (25-node fallback 4--8) static band",hopBalancedDistanceFlows);
  cmd.AddValue("relaySearchDelay","Apply local positive-progress next-hop discovery waiting",relaySearchDelay);
  cmd.AddValue("relaySearchBaseMs","Fixed base component of per-hop relay search waiting",relaySearchBaseMs);
  cmd.AddValue("relaySearchScaleMs","Inverse-candidate component of per-hop relay search waiting",relaySearchScaleMs);
  cmd.AddValue("relaySearchMaxMs","Maximum per-hop relay search waiting",relaySearchMaxMs);
  cmd.AddValue("relaySearchCandidateExponent","Exponent on feasible positive-progress candidate count in relay-search waiting",relaySearchCandidateExponent);
  cmd.AddValue("topologyUpdateCompressionFactor","Fraction of raw neighbour-update burden represented by compressed topology digests",topologyUpdateCompressionFactor);
  cmd.AddValue("aggregatedTopologyDigestAccounting","Use LC-region fixed-capacity topology digest accounting instead of sensor HELLO accounting",aggregatedTopologyDigestAccounting);
  cmd.AddValue("topologyDigestCapacityNodes","Sensor summaries carried by one LC/gateway topology digest",topologyDigestCapacityNodes);
  cmd.Parse(argc,argv);
  gUseFixedControlPolicy=fixedControlPolicy;
  gFixedHelloPeriod=helloSeconds;
  gFixedNegativeRouteTtl=negativeRouteTtlSeconds;
  gFixedRouteTtl=routeTtlSeconds;
  gUseReferenceUpdateAccounting=referenceUpdateAccounting;
  gBufferRouteDiscovery=bufferRouteDiscovery;
  gDisableMcFallback=disableMcFallback;
  gRouteDiscoveryRetryLimit=routeDiscoveryRetryLimit;
  gPacketBufferTimeoutSeconds=packetBufferTimeoutSeconds;
  gLongDistanceFlows=longDistanceFlows;
  if((longDistanceFlows && (balancedDistanceFlows || hopBalancedDistanceFlows)) ||
     (balancedDistanceFlows && hopBalancedDistanceFlows)) {
    std::cerr<<"Only one flow-distance rule may be selected\n"; return 2;
  }
  gBalancedDistanceFlows=balancedDistanceFlows;
  gHopBalancedDistanceFlows=hopBalancedDistanceFlows;
  gUseRelaySearchDelay=relaySearchDelay;
  gRelaySearchBaseMs=relaySearchBaseMs;
  gRelaySearchScaleMs=relaySearchScaleMs;
  gRelaySearchMaxMs=relaySearchMaxMs;
  if(relaySearchCandidateExponent <= 0.) { std::cerr<<"relaySearchCandidateExponent must be positive\n"; return 2; }
  gRelaySearchCandidateExponent=relaySearchCandidateExponent;
  if(topologyUpdateCompressionFactor < 0. || topologyUpdateCompressionFactor > 1.) {
    std::cerr<<"topologyUpdateCompressionFactor must be in [0,1]\n";
    return 2;
  }
  gTopologyUpdateCompressionFactor=topologyUpdateCompressionFactor;
  if(topologyDigestCapacityNodes==0) { std::cerr<<"topologyDigestCapacityNodes must be positive\n"; return 2; }
  gUseAggregatedTopologyDigestAccounting=aggregatedTopologyDigestAccounting;
  gTopologyDigestCapacityNodes=topologyDigestCapacityNodes;
  const auto manifestRecords = ReadScenarioManifest(scenarioManifest);
  std::vector<ScenarioRecord> records = manifestRecords;
  if (records.empty()) {
    records.reserve(runs);
    const uint32_t firstSeed = mode == "run" ? scenarioSeed : baseSeed;
    for (uint32_t i = 0; i < runs; ++i) records.push_back({scenarioId + i, firstSeed + i});
  }
  std::ofstream out(output);if(!out){std::cerr<<"Cannot open "<<output<<'\n';return 2;}
  if(mode=="labels") { WriteLabelHeader(out); for(const auto& record : records) WriteLabelRows(out,record.id,record.seed,nodeCount); }
  else if(mode=="topology-features") {
    WriteTopologyFeatureHeader(out);
    for(const auto& record : records) WriteTopologyFeatureRows(out,record.id,record.seed,nodeCount);
  }
  else if(mode=="network-labels") {
    if(diagnosticOutput.empty()) { std::cerr<<"--diagnosticOutput is required for network-labels\n"; return 2; }
    std::ofstream diagnostic(diagnosticOutput);
    if(!diagnostic) { std::cerr<<"Cannot open "<<diagnosticOutput<<'\n'; return 2; }
    WriteLabelHeader(out);
    diagnostic<<"scenario_id,scenario_seed,node_count,auv_id,generated_packets,delivered_packets,PDR,E2ED_ms,ROR_total,is_oc\n";
    for(const auto& record : records) WriteNetworkLabelRows(out,diagnostic,record.id,record.seed,nodeCount);
  }
  else if(mode=="run") { if(selectedOc>=kNumAuv){std::cerr<<"selectedOc must be 0..3\n";return 2;}WriteRunHeader(out);for(const auto& record : records)WriteRun(out,record.id,record.seed,nodeCount,selectedOc,trace); }
  else { std::cerr<<"Unknown --mode\n";return 2; }
  return 0;
}
