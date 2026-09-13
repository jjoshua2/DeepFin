using System.Runtime.InteropServices;
using System.Text.Json;
using Ceres.Base.DataTypes;
using Ceres.Chess.Positions;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators.Ceres.TPG;
using Ceres.Chess.NetEvaluation.Batch;

// CPU entrypoints only: never construct an evaluator or load a neural model.
if (args.Length != 2) throw new ArgumentException("transport.json output.json required");
if (!BitConverter.IsLittleEndian || Marshal.SizeOf<FP16>() != 2)
    throw new InvalidOperationException("expected two-byte little-endian FP16 layout");
using JsonDocument input = JsonDocument.Parse(File.ReadAllBytes(args[0]));
var histories = input.RootElement.GetProperty("histories");
var values = input.RootElement.GetProperty("values");
if (histories.GetArrayLength() > 144 || values.GetArrayLength() > 144)
    throw new InvalidOperationException("bounded transport exceeded");
var encodings = new List<object>();
foreach (var item in histories.EnumerateArray()) {
    string id = item.GetProperty("id").GetString();
    string fen = item.GetProperty("root_fen").GetString();
    string moves = string.Join(" ", item.GetProperty("moves").EnumerateArray().Select(x => x.GetString()));
    // GetPositions recalculates repetitions on the entire sequence. Never pretruncate.
    var history = PositionWithHistory.FromFENAndMovesUCI(fen, moves);
    var positions = history.GetPositions();
    EncodedPositionWithHistory encoded = default;
    encoded.SetFromSequentialPositions(positions, fillInMissingPlanes: true);
    var record = TPGRecordConverter.ConvertedToTPGRecord(in encoded, includeHistory: true,
        pliesSinceLastPieceMoveBySquare: default, qNegativeBlunders: .03f, qPositiveBlunders: .03f);
    byte[] bytes = new byte[64 * 137];
    record.CopySquares(bytes, 0);
    encodings.Add(new { id, squares_base64 = Convert.ToBase64String(bytes), full_positions = positions.Length });
}
var outputs = new List<object>();
foreach (var item in values.EnumerateArray()) {
    ushort[] a = item.GetProperty("primary_bits").EnumerateArray().Select(x => x.GetUInt16()).ToArray();
    ushort[] b = item.GetProperty("secondary_bits").EnumerateArray().Select(x => x.GetUInt16()).ToArray();
    if (a.Length != 3 || b.Length != 3) throw new InvalidOperationException("WDL triples required");
    FP16[] primary = a.Select(FP16.FromRaw).ToArray();
    FP16[] secondary = b.Select(FP16.FromRaw).ToArray();
    if (!primary.Select(x => x.Value).SequenceEqual(a) || !secondary.Select(x => x.Value).SequenceEqual(b))
        throw new InvalidOperationException("FP16 transport round trip");
    var batch = new PositionEvaluationBatch(isWDL: true, hasM: false, hasUncertaintyV: false,
        hasUnertaintyP: false, hasAction: false, hasValueSecondary: true, hasState: false, numPos: 1,
        valueEvals: primary, valueEvals2: secondary, policyProbs: default, policy2Probs: default,
        actionLogits: default, m: default, uncertaintyV: default, uncertaintyP: default,
        extraStats0: default, extraStats1: default, states: default, activations: null,
        fractionValueFromValue2: .4f, temperatureValue1: .55f, temperatureValue2: 1.5f,
        value1TemperatureUncertaintyScalingFactor: 0f, value2TemperatureUncertaintyScalingFactor: 0f,
        valsAreLogistic: true, probType: PositionEvaluationBatch.PolicyType.LogProbabilities,
        policyAlreadySorted: false, sourceBatchWithValidMoves: null,
        policyTemperature: 1f, policyUncertaintyScalingFactor: 0f,
        fractionPolicyHead2: 0f, policy2BlendLogits: false, policy1Temperature: 1f, policy2Temperature: 1f,
        stats: default);
    FP16[] getters = { batch.GetWinP(0), batch.GetDrawP(0), batch.GetLossP(0), batch.GetV(0),
        batch.GetWin1P(0), batch.GetDraw1P(0), batch.GetLoss1P(0), batch.GetV1(0),
        batch.GetWin2P(0), batch.GetDraw2P(0), batch.GetLoss2P(0), batch.GetV2(0) };
    outputs.Add(new { id = item.GetProperty("id").GetString(), bits = getters.Select(x => x.Value).ToArray(),
        values = getters.Select(x => (float)x).ToArray() });
}
using var target = new FileStream(args[1], FileMode.CreateNew);
JsonSerializer.Serialize(target, new { schema = 1, status = "CPU_UPSTREAM_SEMANTICS_ONLY",
    profile = "explicit_parameterless_options_profile_0.55_1.5_0.4_not_deployment_claim",
    getter_order = new[] { "W", "D", "L", "V", "W1", "D1", "L1", "V1", "W2", "D2", "L2", "V2" },
    encodings, outputs });
