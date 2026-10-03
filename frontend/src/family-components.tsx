import React, { useState, useEffect } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
type Json = Record<string, any>;
export function schemaDefaults(schema: Json): Json {
  return Object.fromEntries(
    Object.entries(schema.properties ?? {})
      .filter(([, p]) => (p as Json).default !== undefined)
      .map(([k, p]) => [k, (p as Json).default]),
  );
}
export function MathPanel({ family, spec }: { family: Json; spec: Json }) {
  const framework =
    family.frameworks.find((f: Json) => f.id === spec.framework_id) ??
    family.frameworks.find(
      (f: Json) =>
        f.controller === spec.controller.id &&
        f.value_function === spec.value_function.id,
    );
  const math = family.math_profiles?.[framework?.id] ?? family.math;
  return (
    <section className="card model-card">
      <h3>{math.title}</h3>
      <p>
        {family.name} / {spec.model.id} /{" "}
        {framework?.label ?? spec.controller.id}
      </p>
      {math.equations.map((s: string) => (
        <div
          className="equation"
          key={s}
          dangerouslySetInnerHTML={{
            __html: katex.renderToString(s, {
              displayMode: true,
              throwOnError: false,
              trust: false,
            }),
          }}
        />
      ))}
      <p>
        公式为当前模型实现的结构摘要，不等同于完整论文推导；模型包内的研究定义与参数是运行依据。
      </p>
      <ul>
        {math.notes.map((s: string) => (
          <li key={s}>{s}</li>
        ))}
      </ul>
      <details>
        <summary>当前数学参数与组件</summary>
        <pre className="json-view">
          {JSON.stringify(
            {
              model: spec.model,
              controller: spec.controller,
              value_function: spec.value_function,
              evaluation: spec.evaluation,
            },
            null,
            2,
          )}
        </pre>
      </details>
    </section>
  );
}
export function BatchBuilder({
  spec,
  plugins,
  onApply,
}: {
  spec: Json;
  plugins: Json[];
  onApply: (batch: Json) => void;
}) {
  const [axes, setAxes] = useState<Json[]>([]),
    [seeds, setSeeds] = useState("0"),
    [scenarios, setScenarios] = useState(""),
    [error, setError] = useState("");
  const options: Json[] = [];
  for (const kind of ["model", "controller", "value_function"]) {
    const p = plugins.find((p) => p.kind === kind && p.id === spec[kind].id);
    for (const [field, def] of Object.entries(
      p?.parameters_schema.properties ?? {},
    )) {
      const d = def as Json;
      const s = d.anyOf?.find((s: Json) => s.type !== "null") ?? d;
      options.push({ path: kind + ".parameters." + field, schema: s });
      if (
        s.type === "array" &&
        Array.isArray(spec[kind].parameters[field] ?? d.default)
      ) {
        (spec[kind].parameters[field] ?? d.default).forEach(
          (_v: any, i: number) =>
            options.push({
              path: kind + ".parameters." + field + "." + i,
              schema: s.items ?? {},
            }),
        );
      }
    }
  }
  const update = (i: number, change: Json) =>
    setAxes(axes.map((a, j) => (j === i ? { ...a, ...change } : a)));
  const apply = () => {
    try {
      const base = structuredClone(spec);
      const values = seeds.split(",").map((s) => Number(s.trim()));
      if (
        values.some((s) => !Number.isInteger(s) || s < 0) ||
        new Set(values).size !== values.length
      )
        throw Error("seed 需要互不重复的非负整数");
      const training = spec.controller.id.startsWith("train_");
      base.execution = {
        ...base.execution,
        policy_seeds: training ? [0] : values,
        ...(training ? { training_seeds: values } : {}),
      };
      base.dataset.scenario_ids = scenarios.trim()
        ? scenarios.split(",").map((s) => s.trim())
        : [];
      const ranges: Json = {},
        sweeps: Json = {};
      for (const a of axes) {
        if (!options.some((o) => o.path === a.path))
          throw Error("扫描参数已随框架切换失效，请重新选择");
        if (ranges[a.path] || sweeps[a.path])
          throw Error("同一参数不能重复扫描");
        if (a.mode === "list") {
          const v = JSON.parse(a.values);
          if (!Array.isArray(v) || !v.length)
            throw Error("离散值须为非空 JSON 数组");
          sweeps[a.path] = v;
        } else
          ranges[a.path] = {
            start: Number(a.start),
            stop: Number(a.stop),
            step: Number(a.step),
            scale: a.mode,
          };
      }
      onApply({
        schema_version: "batch-spec/v1",
        base_spec: base,
        variants: [{}],
        ranges,
        sweeps,
      });
      setError("");
    } catch (e) {
      setError(String(e));
    }
  };
  return (
    <section className="card">
      <h3>参数范围与批实验</h3>
      <p>
        所有模型族使用同一展开协议，参数只来自当前模型包。参数配置不是样本；固定数据中的独立场景才是统计样本。算法比较应复用相同场景。
      </p>
      {axes.map((a, i) => (
        <div className="sweep-row" key={i}>
          <select
            aria-label={"扫描参数 " + (i + 1)}
            value={a.path}
            onChange={(e) => update(i, { path: e.target.value })}
          >
            {options.map((o) => (
              <option key={o.path}>{o.path}</option>
            ))}
          </select>
          <select
            aria-label={"扫描方式 " + (i + 1)}
            value={a.mode}
            onChange={(e) => update(i, { mode: e.target.value })}
          >
            <option value="linear">线性间隔</option>
            <option value="log">对数倍率</option>
            <option value="list">离散 JSON 值</option>
          </select>
          {a.mode === "list" ? (
            <input
              aria-label="离散值"
              value={a.values}
              onChange={(e) => update(i, { values: e.target.value })}
            />
          ) : (
            ["start", "stop", "step"].map((k) => (
              <label key={k}>
                {
                  (
                    { start: "起点", stop: "终点", step: "间隔 / 倍率" } as Json
                  )[k]
                }
                <input
                  type="number"
                  step="any"
                  value={a[k]}
                  onChange={(e) => update(i, { [k]: e.target.value })}
                />
              </label>
            ))
          )}
          <button onClick={() => setAxes(axes.filter((_, j) => j !== i))}>
            移除
          </button>
        </div>
      ))}
      <div className="row">
        <button
          disabled={!options.length}
          onClick={() =>
            setAxes([
              ...axes,
              {
                path: options[0]?.path,
                mode: "linear",
                start: 1,
                stop: 3,
                step: 1,
                values: "[1, 2, 3]",
              },
            ])
          }
        >
          添加扫描参数
        </button>
      </div>
      <div className="two">
        <label>
          {spec.controller.id.startsWith("train_")
            ? "Training seeds"
            : "Policy seeds（不是 scenario seeds）"}
          <input value={seeds} onChange={(e) => setSeeds(e.target.value)} />
          <small>
            逗号分隔；确定性框架通常保留 0，重复 policy seed
            不会增加独立场景样本数。
          </small>
        </label>
        <label>
          场景 ID（留空使用所选 split 的全部场景）
          <input
            value={scenarios}
            onChange={(e) => setScenarios(e.target.value)}
            placeholder="s-10001,s-10002"
          />
          <small>
            scenario seed 来自已冻结数据；可在数据管理中生成 N 个 test 场景。
          </small>
        </label>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
      <button onClick={apply}>应用到批次 JSON</button>
      <p className="muted">
        在下方“预览计划”检查配置数、场景数与
        seed；联合约束不满足时不会提交。可用 variants 对照不同框架。
      </p>
    </section>
  );
}
const periodMetrics = [
  "period_profit",
  "cumulative_profit",
  "period_business_cost",
  "cumulative_business_cost",
  "period_objective",
  "cumulative_objective",
  "pool_before",
  "pool_after",
  "assigned_delta",
  "delivered_delta",
  "assigned_load",
  "delivered_load",
  "idle_vehicles",
  "trip_vehicles",
  "reposition_vehicles",
  "onboard_load",
];
export function AnalysisPanel({
  family,
  runs,
  api,
}: {
  family: Json;
  runs: Json[];
  api: (path: string, body?: unknown) => Promise<any>;
}) {
  const [selected, setSelected] = useState<string[]>([]),
    [kind, setKind] = useState("comparison"),
    [metrics, setMetrics] = useState(["operating_profit"]),
    [x, setX] = useState("model.parameters.empty_cost"),
    [y, setY] = useState("model.parameters.loaded_cost"),
    [width, setWidth] = useState("double"),
    [columns, setColumns] = useState(2),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [result, setResult] = useState<Json | null>(null),
    [saved, setSaved] = useState<Json[]>([]),
    [dataset, setDataset] = useState(""),
    [framework, setFramework] = useState("");
  const refresh = () =>
    api("/analyses?family=" + family.id)
      .then(setSaved)
      .catch((e) => setError(String(e)));
  useEffect(() => {
    refresh();
  }, [family.id]);
  if (!family.analysis_supported)
    return (
      <section className="card">
        <h3>{family.name} · 实验分析</h3>
        <p>
          第一阶段仅开放 single
          的论文分析与导出。当前模型族已使用统一批实验协议，后续可通过本模型包的
          analysis adapter 接入，不与 single 混用指标或 trace。
        </p>
      </section>
    );
  const valid = runs.filter(
    (r) =>
      r.status === "COMPLETED" && !r.spec.controller.id.startsWith("train_"),
  );
  const choices =
    kind === "timeseries"
      ? periodMetrics
      : family.metrics.map((d: Json) => d.key);
  const axes = [
    ...new Set(
      valid.flatMap((r) =>
        ["model", "controller", "value_function"].flatMap((k) =>
          Object.entries(r.spec[k].parameters)
            .filter(([, v]) => typeof v === "number")
            .map(([p]) => k + ".parameters." + p),
        ),
      ),
    ),
  ];
  const perform = async (raw = false) => {
    setBusy(true);
    setError("");
    try {
      const value = await api(
        raw ? "/results/export" : "/analyses",
        raw
          ? { run_ids: selected }
          : {
              run_ids: selected,
              kind,
              metrics,
              x_parameter:
                kind === "sensitivity" || kind === "heatmap" ? x : null,
              y_parameter: kind === "heatmap" ? y : null,
              width,
              columns,
              confidence: 0.95,
              dpi: 600,
              formats: ["pdf", "svg", "png"],
            },
      );
      setResult(value);
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <section className="card">
        <h3>实验选择 · 独立场景样本</h3>
        <p>
          不同算法使用相同冻结场景。不同数据、物理参数、源码或统计口径会分 panel
          展示，不自动合并为样本。选择拟合任务不可当作评估样本。
        </p>
        <div className="row">
          <select
            aria-label="分析数据筛选"
            value={dataset}
            onChange={(e) => setDataset(e.target.value)}
          >
            <option value="">所有数据</option>
            {[...new Set(valid.map((r) => r.spec.dataset.dataset_id))].map(
              (d) => (
                <option key={d}>{d}</option>
              ),
            )}
          </select>
          <select
            aria-label="分析框架筛选"
            value={framework}
            onChange={(e) => setFramework(e.target.value)}
          >
            <option value="">所有框架</option>
            {[
              ...new Set(
                valid.map((r) => r.framework_id ?? r.spec.controller.id),
              ),
            ].map((f) => (
              <option key={f}>{f}</option>
            ))}
          </select>
          <button
            onClick={() =>
              setSelected(
                valid
                  .filter(
                    (r) =>
                      (!dataset || r.spec.dataset.dataset_id === dataset) &&
                      (!framework ||
                        (r.framework_id ?? r.spec.controller.id) === framework),
                  )
                  .map((r) => r.run_id),
              )
            }
          >
            选中筛选结果
          </button>
          <button onClick={() => setSelected([])}>清空</button>
        </div>
        <div className="experiment-selection">
          {runs
            .filter(
              (r) =>
                (!dataset || r.spec.dataset.dataset_id === dataset) &&
                (!framework ||
                  (r.framework_id ?? r.spec.controller.id) === framework),
            )
            .map((r) => (
              <label className="experiment-option" key={r.run_id}>
                <input
                  type="checkbox"
                  disabled={!valid.includes(r)}
                  checked={selected.includes(r.run_id)}
                  onChange={(e) =>
                    setSelected(
                      e.target.checked
                        ? [...selected, r.run_id]
                        : selected.filter((id) => id !== r.run_id),
                    )
                  }
                />
                <span>
                  {r.spec.name} · {r.framework_id ?? r.spec.controller.id}
                  <small>
                    {r.run_id} / {r.spec.dataset.dataset_id} / {r.status} /{" "}
                    {r.metrics?.rows.length ?? 0} 场景 / policy seed{" "}
                    {r.policy_seed ?? "—"}
                  </small>
                  <details>
                    <summary>参数与实验条件</summary>
                    <pre>
                      {JSON.stringify(
                        {
                          model: r.spec.model.parameters,
                          controller: r.spec.controller.parameters,
                          value: r.spec.value_function.parameters,
                        },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                </span>
              </label>
            ))}
        </div>
        <p>
          已选 {selected.length}{" "}
          个运行。失败与缺失不会在分析中被悄悄丢弃；n&lt;2
          仅显示样本，不生成置信区间。
        </p>
      </section>
      <section className="card">
        <h3>论文绘图配置 · 本地 Python</h3>
        <div className="fields">
          <label>
            比较方式
            <select
              value={kind}
              onChange={(e) => {
                setKind(e.target.value);
                setMetrics(
                  e.target.value === "timeseries"
                    ? ["cumulative_profit"]
                    : ["operating_profit"],
                );
              }}
            >
              <option value="comparison">多算法 / 多场景均值、散点与 CI</option>
              <option value="sensitivity">单参数敏感性与置信带</option>
              <option value="heatmap">双参数热力图</option>
              <option value="timeseries">逐 t 动态曲线</option>
              <option value="paired">同场景配对差值</option>
            </select>
          </label>
          <label>
            指标（可多选，最多 6 个 panel 指标）
            <select
              multiple
              value={metrics}
              onChange={(e) =>
                setMetrics(Array.from(e.target.selectedOptions, (o) => o.value))
              }
            >
              {choices.map((m: string) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          {(kind === "sensitivity" || kind === "heatmap") && (
            <label>
              横轴参数
              <select value={x} onChange={(e) => setX(e.target.value)}>
                {axes.map((a) => (
                  <option key={a}>{a}</option>
                ))}
              </select>
            </label>
          )}
          {kind === "heatmap" && (
            <label>
              纵轴参数
              <select value={y} onChange={(e) => setY(e.target.value)}>
                {axes.map((a) => (
                  <option key={a}>{a}</option>
                ))}
              </select>
            </label>
          )}
          <label>
            论文尺寸
            <select value={width} onChange={(e) => setWidth(e.target.value)}>
              <option value="single">单栏 · 3.35 inch</option>
              <option value="double">双栏 · 7 inch</option>
            </select>
          </label>
          <label>
            Panel 列数
            <input
              type="number"
              min={1}
              max={3}
              value={columns}
              onChange={(e) => setColumns(Number(e.target.value))}
            />
          </label>
        </div>
        <p>
          95% pointwise Student-t CI；输出 PDF / SVG / 600 dpi
          PNG。色盲友好配色、不同 marker 和线型支持黑白辨识。augmented objective
          受 value function 影响，只作诊断，不代表运营收益排名。
        </p>
        {error && (
          <div className="error" role="alert">
            {error}
          </div>
        )}
        <div className="row">
          <button
            className="primary"
            disabled={busy || !selected.length}
            onClick={() => perform()}
          >
            生成论文图并保存到本地
          </button>
          <button
            disabled={busy || !selected.length}
            onClick={() => perform(true)}
          >
            导出多实验完整数据
          </button>
        </div>
      </section>
      {result && (
        <section className="card">
          <h3>本地输出 · {result.analysis_id}</h3>
          <p className="path-output">{result.local_path}</p>
          <div className="row">
            {result.artifacts
              ?.filter((n: string) => /^(figure\.|analysis.zip)/.test(n))
              .map((n: string) => (
                <a
                  key={n}
                  download
                  href={
                    "/api/v1/analyses/" + result.analysis_id + "/artifacts/" + n
                  }
                >
                  {n}
                </a>
              ))}
          </div>
          {result.plots?.includes("figure.png") && (
            <img
              className="paper-preview"
              src={
                "/api/v1/analyses/" +
                result.analysis_id +
                "/artifacts/figure.png"
              }
              alt="Python 生成的论文级实验图"
            />
          )}
          <p>
            下载 analysis.zip 后运行 python plotting.py
            可重绘。包内包含绘图配置、原始样本、逐 t 数值 /
            状态记录、统计表及绘图代码。PDF/SVG 是矢量文件，不是网页截图。
          </p>
          <details>
            <summary>统计结果与口径</summary>
            <pre className="json-view">
              {JSON.stringify(result.statistics ?? result.config, null, 2)}
            </pre>
          </details>
        </section>
      )}
      <section className="card">
        <h3>已保存分析</h3>
        {saved.map((a) => (
          <button
            key={a.analysis_id}
            onClick={() =>
              api("/analyses/" + a.analysis_id)
                .then(setResult)
                .catch((e) => setError(String(e)))
            }
          >
            {a.analysis_id} · {a.config.kind}
          </button>
        ))}
      </section>
    </>
  );
}
