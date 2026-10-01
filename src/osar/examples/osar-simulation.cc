// osar-simulation-osar-paper.cc
// OSAR-style simulation, extended to implement:
//  - Markov ON/OFF PU activity (per-node, per-band) using λ (busy->idle) and μ (idle->busy)
//  - Thorp absorption + spreading loss
//  - Ambient noise PSD components (turbulence, shipping, wind/waves, thermal) summed
//  - Small-scale fading (Rayleigh-like) and random environmental loss
//  - SNR -> BER -> PER based packet drops (stochastic)
//  - All important parameters exposed via CommandLine (no hard-coded impairments)
// Build: add to src/osar/examples/CMakeLists.txt
// Run example: ./waf --run "osar-simulation-osar-paper --PuLambda=0.2 --PuMu=0.3 ..."

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/osar-helper.h"
#include <cmath>
#include <random>
#include <vector>
#include <map>
#include <iomanip>

using namespace ns3;
using std::vector;
using std::string;

// ------------------------------
//  Default parameters (overridable via CommandLine)
// ------------------------------
static double g_soundSpeed = 1500.0; // m/s
static double g_pathlossK = 1.5;
static vector<double> g_bandCentersKhz = {13.0, 19.0, 25.0, 31.0, 37.0};
static unsigned int g_subcarriersPerBand = 128;
static double g_bandBwHz = 6000.0;
static double g_packetSizeBytes = 512.0;
static double g_ptW = 5.0; // transmit power (W)
static double g_rxSensitivityDb = -90.0;
static double g_noiseScaleFactor = 1e-18; // maps μPa^2/Hz to effective W/Hz (tunable)
static double g_snrThresholdDb = 5.0;
static double g_simDuration = 10.0;
static double g_sendInterval = 0.1;
static uint32_t g_senderNode = 0;
static uint32_t g_receiverNode = 9;
static double g_neighborRangeM = 380.0;
static double g_surfaceDepthThresh = 5.0;

// PU Markov
static double g_puLambda = 0.2; // busy -> idle rate (1/s)
static double g_puMu = 0.3;     // idle -> busy rate (1/s)
static double g_initialPuBusyProb = 0.3; // initial busy probability

// Ambient factors
static double g_shippingActivity = 0.5; // 0..1
static double g_windSpeed = 5.0; // m/s (for wave noise approx)

// Fading / env loss
static double g_envLossProb = 0.05; // chance of extra env drop per-hop
static double g_fadingMean = 1.0; // mean for exponential fading (|h|^2)

// RNG
static std::mt19937_64 g_rng(1234567);

// ------------------------------
// Helpers: math & conversions
// ------------------------------
double dBToLinear(double db) { return std::pow(10.0, db / 10.0); }
double linearToDb(double lin) { return 10.0 * std::log10(lin); }

// ------------------------------
// Thorp absorption (dB/km)
double AlphaThorp_dB_per_km(double f_khz)
{
  double f2 = f_khz * f_khz;
  double alpha = 0.11 * f2 / (1.0 + f2)
                 + 44.0 * f2 / (4100.0 + f2)
                 + 0.000275 * f2
                 + 0.003;
  return alpha;
}

// ------------------------------
// Transmission loss (dB) TL(d,f) = 10*k*log10(d) + alpha(f) * d(km)
double TransmissionLoss_dB(double d_m, double f_khz)
{
  if (d_m < 1.0) d_m = 1.0;
  double alpha_db_per_km = AlphaThorp_dB_per_km(f_khz);
  double tl = 10.0 * g_pathlossK * std::log10(d_m) + alpha_db_per_km * (d_m / 1000.0);
  return tl;
}

// ------------------------------
// Ambient noise PSD model (dB re 1 μPa^2/Hz)
//  uses four components approximations commonly used in underwater acoustics
//  Nturb (turbulence), Nship (shipping), Nwind/waves, Nthermal
//  Input f_khz
double AmbientNoisePSD_dB(double f_khz, double shippingActivity, double windSpeed)
{
  // f in kHz
  // Turbulence (approx)
  double Nturb = 17.0 - 30.0 * std::log10(f_khz);

  // Shipping (depends on activity s in [0..1]) — approximate form
  double s = std::min(1.0, std::max(0.0, shippingActivity));
  double Nship = 40.0 + 20.0 * (s - 0.5) + 26.0 * std::log10(f_khz) - 60.0 * std::log10(f_khz + 0.03);

  // Wind/waves (windSpeed in m/s approx)
  double w = std::max(0.0, windSpeed);
  // approximate mapping to sea state: use sqrt(w)
  double Nwind = 50.0 + 7.5 * std::sqrt(w) + 20.0 * std::log10(f_khz) - 40.0 * std::log10(f_khz + 0.4);

  // Thermal noise
  double Nthermal = -15.0 + 20.0 * std::log10(f_khz);

  // convert from dB to linear (μPa^2/Hz), sum, convert back to dB
  double Lt = std::pow(10.0, Nturb / 10.0);
  double Ls = std::pow(10.0, Nship / 10.0);
  double Lw = std::pow(10.0, Nwind / 10.0);
  double Lth = std::pow(10.0, Nthermal / 10.0);
  double sum = Lt + Ls + Lw + Lth;
  double totalDb = linearToDb(sum);
  return totalDb;
}

// Given PSD in dB (μPa^2/Hz), convert to an effective noise power (W) over bandwidth B (Hz)
// This requires a calibration factor to map acoustic μPa^2 to W; we expose a scale factor.
double AmbientNoisePower_W(double noisePSD_dB, double B_Hz, double scaleFactor)
{
  // noisePSD_dB = 10*log10( PSD (μPa^2/Hz) )
  // convert to linear μPa^2/Hz
  double psd_uPa2_per_Hz = dBToLinear(noisePSD_dB);
  // convert μPa^2 to Pa^2: 1 μPa = 1e-6 Pa so μPa^2 -> (1e-6)^2 Pa^2 = 1e-12 Pa^2
  double psd_Pa2_per_Hz = psd_uPa2_per_Hz * 1e-12;
  // acoustic intensity to electrical power requires characteristic impedance ~ 1.5e6 kg/(m^2 s)
  // rather than modeling impedance precisely, use a scaleFactor parameter to map to effective Watts
  // scaleFactor default small (tunable)
  double noisePower_W = psd_Pa2_per_Hz * B_Hz * scaleFactor;
  return noisePower_W;
}

// Shannon capacity (bps)
double BandCapacity_bps(double Pr_W, double B_Hz, double noiseW)
{
  double snr = Pr_W / (noiseW + 1e-30);
  if (snr <= 0) return 0.0;
  double C = B_Hz * std::log2(1.0 + snr);
  return C;
}

// ------------------------------
// Simulation state & counters
// ------------------------------
struct Metrics
{
  uint64_t txPackets = 0;
  uint64_t rxPackets = 0;
  double delaySum = 0.0; // seconds
  uint64_t controlPackets = 0; // sensing broadcasts, etc.
  double energyConsumedJ = 0.0; // J
} metrics;

struct NodeSenseState
{
  vector<bool> bandIdle;
  NodeSenseState() : bandIdle(g_bandCentersKhz.size(), true) {}
};

vector<NodeSenseState> nodeSenseState;

// ------------------------------
// PU Markov state per-node per-band
struct PUState {
  bool busy;
  double nextTransitionTime; // simulator time of next state flip
};
vector<vector<PUState>> puState;

// RNG helper distributions
std::uniform_real_distribution<double> urand01(0.0, 1.0);

// Initialize PU states (call after nodes created)
void InitializePUStates(uint32_t nNodes)
{
  puState.clear();
  puState.resize(nNodes);
  for (uint32_t i=0;i<nNodes;++i)
  {
    puState[i].resize(g_bandCentersKhz.size());
    for (size_t b=0;b<g_bandCentersKhz.size();++b)
    {
      // initial busy with given probability
      bool busy = (urand01(g_rng) < g_initialPuBusyProb);
      puState[i][b].busy = busy;
      // schedule first transition using exponential with appropriate rate
      double rate = busy ? g_puLambda : g_puMu;
      if (rate <= 0.0) puState[i][b].nextTransitionTime = std::numeric_limits<double>::infinity();
      else {
        std::exponential_distribution<double> ed(rate);
        puState[i][b].nextTransitionTime = Simulator::Now().GetSeconds() + ed(g_rng);
      }
    }
  }
}

// Update PU markov states according to Simulator time; update nodeSenseState accordingly
void UpdatePUStates(NodeContainer &nodes)
{
  double now = Simulator::Now().GetSeconds();
  for (uint32_t i=0;i<nodes.GetN();++i)
  {
    for (size_t b=0;b<g_bandCentersKhz.size();++b)
    {
      PUState &st = puState[i][b];
      // process possible multiple transitions if large time-step
      while (now >= st.nextTransitionTime)
      {
        // flip state
        st.busy = !st.busy;
        // schedule next transition
        double rate = st.busy ? g_puLambda : g_puMu;
        if (rate <= 0.0) { st.nextTransitionTime = std::numeric_limits<double>::infinity(); break; }
        std::exponential_distribution<double> ed(rate);
        double dt = ed(g_rng);
        st.nextTransitionTime += dt;
      }
      nodeSenseState[i].bandIdle[b] = !st.busy;
    }
  }
}

// ------------------------------
// Mobility helpers
Vector GetNodePositionVec(Ptr<Node> n)
{
  Ptr<MobilityModel> mm = n->GetObject<MobilityModel>();
  if (!mm) return Vector(0,0,0);
  return mm->GetPosition();
}
double DistanceM(Ptr<Node> a, Ptr<Node> b)
{
  Vector pa = GetNodePositionVec(a);
  Vector pb = GetNodePositionVec(b);
  double dx = pa.x - pb.x;
  double dy = pa.y - pb.y;
  double dz = pa.z - pb.z;
  return std::sqrt(dx*dx + dy*dy + dz*dz);
}
double DepthOfNode(Ptr<Node> n)
{
  Vector p = GetNodePositionVec(n);
  return std::max(0.0, p.z);
}

// ------------------------------
// Neighbor discovery
vector<uint32_t> FindNeighbors(NodeContainer &nodes, uint32_t idx, double rangeM)
{
  vector<uint32_t> nbrs;
  Ptr<Node> src = nodes.Get(idx);
  for (uint32_t j=0;j<nodes.GetN();++j)
  {
    if (j==idx) continue;
    Ptr<Node> nd = nodes.Get(j);
    double d = DistanceM(src, nd);
    if (d <= rangeM) nbrs.push_back(j);
  }
  return nbrs;
}

// Check common idle band
bool HaveCommonIdleBand(uint32_t i, uint32_t j, int &bandIndex)
{
  for (size_t b=0;b<g_bandCentersKhz.size();++b)
  {
    if (nodeSenseState[i].bandIdle[b] && nodeSenseState[j].bandIdle[b])
    {
      bandIndex = (int)b; return true;
    }
  }
  bandIndex = -1; return false;
}

// ------------------------------
// OSAR helpers: PiD, Ni, metric
vector<uint32_t> BuildPiD(NodeContainer &nodes, uint32_t idx, uint32_t D_idx, double neighborRangeM)
{
  vector<uint32_t> piD;
  Ptr<Node> ni = nodes.Get(idx);
  double depth_i = DepthOfNode(ni);
  vector<uint32_t> nbrs = FindNeighbors(nodes, idx, neighborRangeM);
  for (uint32_t n : nbrs)
  {
    Ptr<Node> nj = nodes.Get(n);
    double depth_j = DepthOfNode(nj);
    if (depth_j < depth_i - 1e-6) piD.push_back(n);
  }
  return piD;
}

vector<uint32_t> BuildNi(NodeContainer &nodes, uint32_t idx, const vector<uint32_t> &piD)
{
  vector<uint32_t> Ni;
  for (uint32_t j : piD)
  {
    int band=-1;
    if (HaveCommonIdleBand(idx, j, band)) Ni.push_back(j);
  }
  return Ni;
}

double AverageVerticalAdvance(NodeContainer &nodes, uint32_t idx, const vector<uint32_t> &piD)
{
  Ptr<Node> ni = nodes.Get(idx);
  double depth_i = DepthOfNode(ni);
  if (piD.empty()) return 0.0;
  double sum=0.0;
  for (uint32_t j: piD) { Ptr<Node> nj=nodes.Get(j); sum += std::max(0.0, depth_i - DepthOfNode(nj)); }
  return sum / (double)piD.size();
}

double ComputeTDiJch(NodeContainer &nodes, uint32_t i, uint32_t j, int bandIndex, uint32_t D_idx, const vector<uint32_t> &piD)
{
  Ptr<Node> ni = nodes.Get(i), nj = nodes.Get(j);
  double depth_i = DepthOfNode(ni), depth_j = DepthOfNode(nj);
  double Depth_iD = depth_i;
  double Dij_vert = std::max(0.0, depth_i - depth_j);
  if (Dij_vert <= 0.0) return 1e18;
  double avgAdvance = AverageVerticalAdvance(nodes, i, piD);
  if (avgAdvance <= 1e-9) avgAdvance = Dij_vert;
  double N_ij_Hop = std::max((Depth_iD / avgAdvance), 1.0);

  double d3 = DistanceM(ni, nj);
  double fkhz = g_bandCentersKhz[bandIndex];
  double tl_db = TransmissionLoss_dB(d3, fkhz);
  double Pt_db = linearToDb(g_ptW);
  double Pr_db = Pt_db - tl_db;
  double Pr_W = dBToLinear(Pr_db);

  // noise for this band
  double noisePSD_dB = AmbientNoisePSD_dB(fkhz, g_shippingActivity, g_windSpeed);
  double noiseW = AmbientNoisePower_W(noisePSD_dB, g_bandBwHz, g_noiseScaleFactor);

  double Cbps = BandCapacity_bps(Pr_W, g_bandBwHz, noiseW);
  if (Cbps <= 1.0) return 1e18;

  double packetBits = g_packetSizeBytes * 8.0;
  double term_tx = packetBits / Cbps;
  double PDi_j = Dij_vert / g_soundSpeed;

  // optional depth weight (aligns with desire to prefer shallower)
  double depthWeight = 1.0 / (1.0 + DepthOfNode(nj));

  return (term_tx + PDi_j * N_ij_Hop) * depthWeight;
}

double ComputeActualDelayForMetrics(NodeContainer &nodes, uint32_t i, uint32_t j, int bandIndex)
{
  Ptr<Node> ni = nodes.Get(i), nj = nodes.Get(j);
  double d3 = DistanceM(ni, nj);
  double fkhz = g_bandCentersKhz[bandIndex];
  double tl_db = TransmissionLoss_dB(d3, fkhz);
  double Pt_db = linearToDb(g_ptW);
  double Pr_db = Pt_db - tl_db;
  double Pr_W = dBToLinear(Pr_db);
  double noisePSD_dB = AmbientNoisePSD_dB(fkhz, g_shippingActivity, g_windSpeed);
  double noiseW = AmbientNoisePower_W(noisePSD_dB, g_bandBwHz, g_noiseScaleFactor);
  double Cbps = BandCapacity_bps(Pr_W, g_bandBwHz, noiseW);
  if (Cbps <= 1.0) return 1e9;
  double packetBits = g_packetSizeBytes * 8.0;
  double txTime = packetBits / Cbps;
  double prop = d3 / g_soundSpeed;
  return txTime + prop;
}

// ------------------------------
// Packet error model helpers
// ------------------------------
bool PacketDropByChannel(NodeContainer &nodes, uint32_t current, uint32_t bestNbr, int bestBand)
{
  Ptr<Node> ni = nodes.Get(current), nj = nodes.Get(bestNbr);
  double d3 = DistanceM(ni, nj);
  double fkhz = g_bandCentersKhz[bestBand];
  double tl_db = TransmissionLoss_dB(d3, fkhz);
  double Pt_db = linearToDb(g_ptW);
  double Pr_db = Pt_db - tl_db;

  // small-scale fading (Rayleigh-like: |h|^2 exponential with mean 1)
  std::exponential_distribution<double> expf(g_fadingMean);
  double fading = expf(g_rng); // >0
  double Pr_W = dBToLinear(Pr_db) * fading;

  // noise
  double noisePSD_dB = AmbientNoisePSD_dB(fkhz, g_shippingActivity, g_windSpeed);
  double noiseW = AmbientNoisePower_W(noisePSD_dB, g_bandBwHz, g_noiseScaleFactor);

  double snr_lin = Pr_W / (noiseW + 1e-30);
  double snr_db = linearToDb(snr_lin);

  // basic BER model (approx for BPSK/QPSK-ish underwater)
  // BER ≈ 0.5 * exp(-snr_lin/2)
  double ber = 0.5 * std::exp(-snr_lin / 2.0);
  double packetBits = g_packetSizeBytes * 8.0;
  double per = 1.0 - std::pow(1.0 - ber, packetBits);

  // environmental random loss
  std::bernoulli_distribution envDrop(g_envLossProb);
  bool envLost = envDrop(g_rng);

  // final probabilistic drop
  double totalProb = per + (envLost ? 0.1 : 0.0); // small extra penalty if env lost event occurred
  if (totalProb > 0.9999) totalProb = 0.9999;
  std::bernoulli_distribution finalDrop(totalProb);
  bool drop = finalDrop(g_rng);

  // debug info (optional): uncomment to print per-hop SNR/BER/PER
  // std::cout << "  [PHY] SNR=" << snr_db << " dB, BER=" << ber << ", PER=" << per << ", env=" << envLost << "\n";

  return drop;
}

// ------------------------------
// Simulated send (OSAR) with all losses
bool SimulateSendPacket_OSAR(NodeContainer &nodes, uint32_t srcIdx, uint32_t dstIdx, double sendTime)
{
  metrics.txPackets++;
  vector<uint32_t> srcNbrs = FindNeighbors(nodes, srcIdx, g_neighborRangeM);
  metrics.controlPackets += srcNbrs.size();

  uint32_t current = srcIdx;
  int hopCount = 0;
  const int MAX_HOPS = 50;

  while (hopCount < MAX_HOPS)
  {
    Ptr<Node> curNode = nodes.Get(current);
    double curDepth = DepthOfNode(curNode);
    if (curDepth <= g_surfaceDepthThresh)
    {
      metrics.rxPackets++;
      return true;
    }

    vector<uint32_t> piD = BuildPiD(nodes, current, dstIdx, g_neighborRangeM);
    vector<uint32_t> Ni = BuildNi(nodes, current, piD);
    if (Ni.empty()) return false;

    double bestMetric = 1e18;
    int bestNbr=-1, bestBand=-1;
    for (uint32_t cand: Ni)
    {
      for (size_t b=0;b<g_bandCentersKhz.size();++b)
      {
        if (!(nodeSenseState[current].bandIdle[b] && nodeSenseState[cand].bandIdle[b])) continue;
        double T = ComputeTDiJch(nodes, current, cand, (int)b, dstIdx, piD);
        if (T < bestMetric) { bestMetric = T; bestNbr = (int)cand; bestBand = (int)b; }
      }
    }
    if (bestNbr == -1) return false;

    // compute SNR and decide if PHY drop occurs
    // quick SNR estimate (no fading) for threshold check
    Ptr<Node> ni = nodes.Get(current), nj = nodes.Get(bestNbr);
    double d3 = DistanceM(ni, nj);
    double fkhz = g_bandCentersKhz[bestBand];
    double tl_db = TransmissionLoss_dB(d3, fkhz);
    double Pt_db = linearToDb(g_ptW);
    double Pr_db = Pt_db - tl_db;
    double Pr_W_noFading = dBToLinear(Pr_db);
    double noisePSD_dB = AmbientNoisePSD_dB(fkhz, g_shippingActivity, g_windSpeed);
    double noiseW = AmbientNoisePower_W(noisePSD_dB, g_bandBwHz, g_noiseScaleFactor);
    double snr_lin_noFading = Pr_W_noFading / (noiseW + 1e-30);
    double snr_db_noFading = linearToDb(snr_lin_noFading);
    if (snr_db_noFading < g_snrThresholdDb) return false;

    double linkDelay = ComputeActualDelayForMetrics(nodes, current, bestNbr, bestBand);
    if (linkDelay > 1e6) return false;

    // apply stochastic channel errors (fading + BER -> PER)
    bool dropped = PacketDropByChannel(nodes, current, bestNbr, bestBand);
    if (dropped) return false;

    // account energy & delay
    double Cbps = BandCapacity_bps(Pr_W_noFading, g_bandBwHz, noiseW);
    double packetBits = g_packetSizeBytes * 8.0;
    double txTime = packetBits / std::max(Cbps, 1.0);
    metrics.energyConsumedJ += g_ptW * txTime;
    metrics.delaySum += linkDelay;

    // print hop info
    std::cout << "Hop " << hopCount << ": node " << current << " -> node " << bestNbr
              << " (band " << bestBand << ", d=" << d3 << " m)\n";
    std::cout << "     depth(current)=" << DepthOfNode(ni)
              << " depth(next)=" << DepthOfNode(nj)
              << " verticalAdvance=" << (DepthOfNode(ni) - DepthOfNode(nj))
              << " remainingDepth=" << DepthOfNode(nj) << "\n";

    current = (uint32_t)bestNbr;
    hopCount++;
  }

  return false;
}

// ------------------------------
// Sending schedule & sensing orchestration
void SchedulePeriodicSend(Ptr<Node> node, NodeContainer &nodes, double startTime)
{
  double t = startTime;
  while (t < g_simDuration - 1e-6)
  {
    Simulator::Schedule(Seconds(t), [&nodes]() {
      // update Markov PU states and derive sensing before each send
      UpdatePUStates(const_cast<NodeContainer&>(nodes));
      bool ok = SimulateSendPacket_OSAR(const_cast<NodeContainer&>(nodes), g_senderNode, g_receiverNode, Simulator::Now().GetSeconds());
      (void)ok;
    });
    t += g_sendInterval;
  }
}

// ------------------------------
// Main
int main(int argc, char *argv[])
{
  CommandLine cmd;
  // allow overriding parameters (none are forced hard-coded)
  cmd.AddValue("ptW", "Transmit power (W)", g_ptW);
  cmd.AddValue("sendInterval", "Inter-packet interval (s)", g_sendInterval);
  cmd.AddValue("simDuration", "Simulation duration (s)", g_simDuration);
  cmd.AddValue("senderNode", "Sender node index", g_senderNode);
  cmd.AddValue("receiverNode", "Receiver/surface node index", g_receiverNode);
  cmd.AddValue("neighborRange", "Neighbor discovery range (m)", g_neighborRangeM);
  cmd.AddValue("puLambda", "PU busy->idle rate (1/s)", g_puLambda);
  cmd.AddValue("puMu", "PU idle->busy rate (1/s)", g_puMu);
  cmd.AddValue("initialPuBusyProb", "Initial busy prob for PU states", g_initialPuBusyProb);
  cmd.AddValue("shipping", "Shipping activity (0..1)", g_shippingActivity);
  cmd.AddValue("windSpeed", "Wind speed (m/s)", g_windSpeed);
  cmd.AddValue("noiseScale", "Noise PSD -> W scale factor", g_noiseScaleFactor);
  cmd.AddValue("envLossProb", "Environmental random loss probability", g_envLossProb);
  cmd.AddValue("snrThresholdDb", "SNR threshold (dB)", g_snrThresholdDb);
  cmd.Parse(argc, argv);

  // basic logging
  LogComponentEnable("OsarHelper", LOG_LEVEL_INFO);
  std::cout << "=== OSAR Simulation (paper-like PHY & PU) Start ===\n";

  NodeContainer nodes;
  nodes.Create(20);

  MobilityHelper mobility;
  mobility.SetPositionAllocator("ns3::GridPositionAllocator",
                                "MinX", DoubleValue(0.0),
                                "MinY", DoubleValue(0.0),
                                "DeltaX", DoubleValue(200.0),
                                "DeltaY", DoubleValue(200.0),
                                "GridWidth", UintegerValue(5),
                                "LayoutType", StringValue("RowFirst"));
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.Install(nodes);

  // depth assignment: node 0 deep, nodes 1-4 mid, nodes 5-9 near surface
  for (uint32_t i=0;i<nodes.GetN();++i)
  {
    Ptr<Node> n = nodes.Get(i);
    Ptr<MobilityModel> mm = n->GetObject<MobilityModel>();
    Vector p = mm->GetPosition();
    if (i==0) p.z = 500.0;
    else if (i < 5) p.z = 400.0 - (i * 75.0);
    else p.z = (i - 5) * 5.0;
    mm->SetPosition(p);
  }

  // print node positions
  std::cout << "Node positions (x,y,z):\n";
  for (uint32_t i=0;i<nodes.GetN();++i)
  {
    Ptr<Node> n = nodes.Get(i);
    Ptr<MobilityModel> mm = n->GetObject<MobilityModel>();
    Vector p = mm->GetPosition();
    std::cout << " node " << i << ": (" << p.x << ", " << p.y << ", " << p.z << ")\n";
  }

  OsarHelper osar;
  osar.Install(nodes);

  nodeSenseState.assign(nodes.GetN(), NodeSenseState());
  InitializePUStates(nodes.GetN());

  // schedule repeated sensing + sends
  SchedulePeriodicSend(nodes.Get(g_senderNode), nodes, 1.0);

  Simulator::Stop(Seconds(g_simDuration));
  Simulator::Run();

  double pdr = (metrics.txPackets > 0) ? (double)metrics.rxPackets / (double)metrics.txPackets * 100.0 : 0.0;
  double avgDelay = (metrics.rxPackets > 0) ? metrics.delaySum / (double)metrics.rxPackets : 0.0;
  double energyEff = (metrics.energyConsumedJ > 0.0) ? (double)metrics.rxPackets / metrics.energyConsumedJ : 0.0;
  double overheadPct = (metrics.rxPackets > 0) ?
                     ((double)metrics.controlPackets / (double)metrics.rxPackets) * 100.0 :
                     0.0;



  std::cout << std::fixed << std::setprecision(6);
  std::cout << "\n=== Evaluation Metrics (OSAR-mode) ===\n";
  std::cout << "Packets Sent (attempted): " << metrics.txPackets << "\n";
  std::cout << "Packets Received (delivered): " << metrics.rxPackets << "\n";
  std::cout << "Packet Delivery Ratio (PDR): " << pdr << " %\n";
  std::cout << "Average End-to-End Delay: " << avgDelay << " s\n";
  std::cout << "Energy Consumed: " << metrics.energyConsumedJ << " J\n";
  std::cout << "Energy Efficiency: " << energyEff << " packets/J\n";
  std::cout << "Routing Overhead (control pkts per data pkt %): " << overheadPct << " %\n";
  std::cout << "Simulation Time (sim): " << Simulator::Now().GetSeconds() << " s\n";
  std::cout << "==========================================\n";

  Simulator::Destroy();
  return 0;
}

