import React, { useEffect, useState, useRef } from "react";
import { createRoot } from "react-dom/client";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
} from "recharts";
import "./style.css";
import {
  AnalysisPanel,
  BatchBuilder,
  MathPanel,
  schemaDefaults,
} from "./family-components";

type Json = Record<string, any>;
type Plugin = {
  kind: string;
  id: string;
  version: string;
  parameters_schema: Json;
};
let token = "";
async function requestApi(path: string, body?: unknown): Promise<any> {
  if (body !== undefined && !token)
    token = (await requestApi("/session")).write_token;
  const response = await fetch("/api/v1" + path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "X-Workspace-Token": token },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw Error(JSON.stringify(data.error ?? data));
  return data;
}
const pretty = (x: unknown) => JSON.stringify(x, null, 2);
const pages = [
  "总览",
  "数据管理",
  "实验编辑器",
  "队列",
  "运行详情",
  "实验比较",
  "回放",
  "BHH 分析",
  "复现",
  "数学模型",
  "实验分析",
  "诊断",
];
function defaultSpec(meta?: Json): Json {
  if (meta) {
    const b = defaultSpec(),
      f = meta.frameworks[0],
      m = f.model ?? meta.models[0];
    return {
      ...b,
      family: meta.id,
      framework_id: f.id,
      name: meta.name + " · " + f.label,
      dataset: { ...b.dataset, dataset_id: meta.id + "-demo", revision: "" },
      model: { id: m, version: "1", parameters: {} },
      controller: { id: f.controller, version: "1", parameters: {} },
      value_function: { id: f.value_function, version: "1", parameters: {} },
      solver: { ...b.solver, backend: meta.backends[m] },
      evaluation: {
        ...b.evaluation,
        accounting_version: meta.accounting,
        metrics: meta.metrics.map((m: Json) => m.key),
        information_set: f.information_set ?? "online",
      },
    };
  }

  return {
    schema_version: "experiment-spec/v1",
    name: "单层匹配 · 基线",
    dataset: {
      dataset_id: "matching-demo",
      revision: "",
      split: "test",
      scenario_ids: [],
    },
    model: { id: "single_level_matching", version: "1", parameters: {} },
    controller: { id: "myopic", version: "1", parameters: {} },
    value_function: { id: "zero", version: "1", parameters: {} },
    solver: {
      backend: "gurobi",
      parameters: { time_limit_seconds: 30, mip_gap: 0, threads: 1, seed: 0 },
    },
    evaluation: {
      information_set: "online",
      accounting_version: "legacy-assignment-v1",
      terminal_policy: "report_pending",
      warmup_periods: 0,
      metrics: ["operating_profit", "assigned", "delivered", "pending"],
    },
    execution: { policy_seeds: [0], save_trace: true, timeout_seconds: 300 },
  };
}
function JsonView({ value }: { value: unknown }) {
  return <pre className="json-view">{pretty(value)}</pre>;
}
function ParameterFields({
  schema,
  value,
  onChange,
}: {
  schema: Json;
  value: Json;
  onChange: (value: Json) => void;
}) {
  return (
    <div className="fields">
      {Object.entries(schema.properties ?? {}).map(([key, item]) => {
        const definition = item as Json;
        const p =
          definition.anyOf?.find((x: Json) => x.type !== "null") ?? definition;
        const current = value[key] ?? definition.default;
        return (
          <label key={key}>
            <span>{key}</span>
            <small>
              {p.description ??
                `${p.minimum ?? p.exclusiveMinimum ?? ""}${p.maximum !== undefined ? " … " + p.maximum : ""}`}
            </small>
            {p.type === "boolean" ? (
              <select
                value={String(current ?? false)}
                onChange={(e) =>
                  onChange({ ...value, [key]: e.target.value === "true" })
                }
              >
                <option>true</option>
                <option>false</option>
              </select>
            ) : p.enum ? (
              <select
                value={current}
                onChange={(e) => onChange({ ...value, [key]: e.target.value })}
              >
                {p.enum.map((x: string) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            ) : p.type === "number" || p.type === "integer" ? (
              <input
                type="number"
                value={current ?? ""}
                step={p.type === "integer" ? 1 : "any"}
                onChange={(e) =>
                  onChange({ ...value, [key]: Number(e.target.value) })
                }
              />
            ) : p.type === "array" || p.type === "object" ? (
              <textarea
                key={key + pretty(current)}
                defaultValue={pretty(current ?? null)}
                onBlur={(e) => {
                  try {
                    onChange({ ...value, [key]: JSON.parse(e.target.value) });
                    e.target.setCustomValidity("");
                  } catch {
                    e.target.setCustomValidity("需要合法 JSON");
                  }
                }}
              />
            ) : (
              <input
                value={current ?? ""}
                onChange={(e) => onChange({ ...value, [key]: e.target.value })}
              />
            )}
          </label>
        );
      })}
    </div>
  );
}
function StudyHeatmap({ runs }: { runs: Json[] }) {
  const batches = [
    ...new Set(
      runs
        .filter(
          (r) => r.spec.model.id === "bhh_steady" && r.status === "COMPLETED",
        )
        .map((r) => r.batch_id),
    ),
  ];
  const [batch, setBatch] = useState("");
  const selected = batch || batches[0];
  const records = runs.filter(
    (r) =>
      r.batch_id === selected && r.spec.model.id === "bhh_steady" && r.metrics,
  );
  const taus = [
    ...new Set(records.map((r) => r.spec.model.parameters.tau)),
  ].sort((a, b) => a - b);
  const rates = [
    ...new Set(records.map((r) => r.spec.model.parameters.demand_rate)),
  ].sort((a, b) => a - b);
  const costs = records
    .map((r) => r.metrics.rows[0]?.metrics.business_cost)
    .filter((x) => typeof x === "number");
  const min = Math.min(...costs),
    max = Math.max(...costs);
  return (
    <section className="card">
      <h3>τ × Λ 参数扫描 · 理论单位成本</h3>
      <p>同批次参数研究，不是在线策略排名。每个单元格来自实际运行。</p>
      <select
        aria-label="扫描批次"
        value={selected ?? ""}
        onChange={(e) => setBatch(e.target.value)}
      >
        {batches.map((b) => (
          <option key={b}>{b}</option>
        ))}
      </select>
      {records.length ? (
        <table>
          <thead>
            <tr>
              <th>τ / Λ</th>
              {rates.map((rate) => (
                <th key={rate}>{rate}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {taus.map((tau) => (
              <tr key={tau}>
                <th>{tau}</th>
                {rates.map((rate) => {
                  const r = records.find(
                    (r) =>
                      r.spec.model.parameters.tau === tau &&
                      r.spec.model.parameters.demand_rate === rate,
                  );
                  const cost = r?.metrics.rows[0]?.metrics.business_cost;
                  return (
                    <td
                      key={rate}
                      title={r?.run_id}
                      style={{
                        background:
                          typeof cost === "number"
                            ? `rgba(19,133,121,${0.1 + (0.55 * (cost - min)) / Math.max(1, max - min)})`
                            : undefined,
                      }}
                    >
                      {typeof cost === "number" ? cost.toFixed(3) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p>运行 configs/batches/bhh-tau-demand.json 后显示热力图。</p>
      )}
    </section>
  );
}
function Network({ network, state }: { network: Json; state?: Json }) {
  const nodes: string[] = network.nodes ?? [],
    positions = Object.fromEntries(
      nodes.map((n, i) => [
        n,
        {
          x: 230 + 180 * Math.cos((i / nodes.length) * 2 * Math.PI),
          y: 180 + 135 * Math.sin((i / nodes.length) * 2 * Math.PI),
        },
      ]),
    );
  return (
    <>
      <svg viewBox="0 0 460 360" role="img" aria-label="抽象网络与车辆位置">
        {(network.edges ?? []).map(([a, b, w]: any[], i: number) => (
          <g key={i}>
            <line
              x1={positions[a]?.x}
              y1={positions[a]?.y}
              x2={positions[b]?.x}
              y2={positions[b]?.y}
              stroke="#c9d5e1"
              strokeWidth="2"
            />
            <text
              x={(positions[a]?.x + positions[b]?.x) / 2}
              y={(positions[a]?.y + positions[b]?.y) / 2}
              fontSize="10"
              fill="#8293a7"
            >
              {w}
            </text>
          </g>
        ))}
        {nodes.map((n) => (
          <g key={n}>
            <circle
              cx={positions[n].x}
              cy={positions[n].y}
              r="21"
              fill="#e7f1f1"
              stroke="#138579"
            />
            <text
              x={positions[n].x}
              y={positions[n].y + 5}
              textAnchor="middle"
              fill="#12665f"
            >
              {n}
            </text>
            <text
              x={positions[n].x}
              y={positions[n].y + 38}
              textAnchor="middle"
              fontSize="10"
            >
              {state?.vehicles?.filter((v: Json) => v.hub === n).length ?? 0} 辆
            </text>
          </g>
        ))}
      </svg>
      <small>抽象布局 · 非地理地图；位置为已提交状态，不代表规划到达。</small>
    </>
  );
}
function Workspace({
  meta,
  families,
  onFamily,
}: {
  meta: Json;
  families: Json[];
  onFamily: (id: string) => void;
}) {
  const api = async (path: string, body?: any) => {
    if (
      body !== undefined &&
      (path === "/batches" ||
        path.startsWith("/experiments/") ||
        path === "/debug/step")
    ) {
      body = structuredClone(body);
      const base = body.base_spec ?? body.spec ?? body;
      if (!meta.models.includes(base.model.id))
        throw Error("JSON 模型不属于当前工作区，请先切换模型族");
      base.family = meta.id;
      for (const v of body.variants ?? []) {
        if (
          (v.model?.id && !meta.models.includes(v.model.id)) ||
          (v.family && v.family !== meta.id)
        )
          throw Error("批次不能跨模型族混合执行");
        v.family = meta.id;
      }
    }
    return requestApi(path, body);
  };

  const [page, setPage] = useState("总览"),
    [datasets, setDatasets] = useState<Json[]>([]),
    [runs, setRuns] = useState<Json[]>([]),
    [plugins, setPlugins] = useState<Plugin[]>([]),
    [spec, setSpec] = useState<Json>(() => {
      try {
        return (
          JSON.parse(localStorage.getItem("spec:" + meta.id) ?? "null") ??
          defaultSpec(meta)
        );
      } catch {
        return defaultSpec(meta);
      }
    }),
    [raw, setRaw] = useState(""),
    [error, setError] = useState(""),
    [notice, setNotice] = useState(""),
    [result, setResult] = useState<any>(null),
    [selected, setSelected] = useState(""),
    [detail, setDetail] = useState<Json | null>(null),
    [trace, setTrace] = useState<Json[]>([]),
    [period, setPeriod] = useState(0),
    [scenario, setScenario] = useState(0),
    [capabilities, setCapabilities] = useState<Json | null>(null),
    [comparisonIds, setComparisonIds] = useState<string[]>([]),
    [preview, setPreview] = useState<Json | null>(null),
    [generation, setGeneration] = useState<Json>({
      ...schemaDefaults(meta.generation_schema),
      dataset_id: meta.id + "-demo",
      seeds: [10001, 10002, 10003, 10004],
      splits: ["test"],
      horizon: 6,
      orders_per_step: 2,
    }),
    [importText, setImportText] = useState(""),
    [importFormat, setImportFormat] = useState("csv");
  const busy = useRef(false);
  const act = async (fn: () => Promise<unknown>, message = "操作完成") => {
    if (busy.current) return null;
    busy.current = true;
    setError("");
    try {
      const value = await fn();
      setResult(value);
      setNotice(message);
      return value;
    } catch (e) {
      setError(String(e));
      return null;
    } finally {
      busy.current = false;
    }
  };
  const refresh = async () => {
    try {
      const [d, r] = await Promise.all([
        api("/datasets?family=" + meta.id),
        api("/runs?family=" + meta.id),
      ]);
      setDatasets(d);
      setRuns(r);
    } catch (e) {
      setError(String(e));
    }
  };
  useEffect(() => {
    refresh();
    api("/plugins?family=" + meta.id)
      .then((items: Plugin[]) => {
        setPlugins(items);
        setSpec((previous: Json) => {
          const result = { ...previous };
          for (const kind of ["model", "controller", "value_function"]) {
            const p = items.find(
              (p) => p.kind === kind && p.id === previous[kind].id,
            );
            result[kind] = {
              ...previous[kind],
              parameters: {
                ...schemaDefaults(p?.parameters_schema ?? {}),
                ...previous[kind].parameters,
              },
            };
          }
          return result;
        });
      })
      .catch((e) => setError(String(e)));
    api("/capabilities")
      .then(setCapabilities)
      .catch((e) => setError(String(e)));
    const interval = setInterval(refresh, 3000);
    return () => clearInterval(interval);
  }, []);
  useEffect(() => {
    setRaw(pretty(spec));
    localStorage.setItem("spec:" + meta.id, pretty(spec));
  }, [spec]);
  useEffect(() => {
    if (!selected) return;
    api("/runs/" + selected)
      .then(setDetail)
      .catch((e) => setError(String(e)));
    api("/runs/" + selected + "/artifacts/trace.json")
      .then(setTrace)
      .catch(() => setTrace([]));
  }, [selected, runs]);
  const chooseData = (d: Json) =>
    setSpec({
      ...spec,
      dataset: {
        dataset_id: d.dataset_id,
        revision: d.revision,
        split: spec.controller.id.startsWith("train_") ? "train" : "test",
        scenario_ids: [],
      },
    });
  const selectFramework = (f: Json, training = false) => {
    const model = f.model ?? meta.models[0];
    setSpec({
      ...spec,
      family: meta.id,
      framework_id: f.id,
      model: {
        id: model,
        version: "1",
        parameters:
          spec.model.id === model
            ? spec.model.parameters
            : schemaDefaults(
                plugins.find((p) => p.kind === "model" && p.id === model)
                  ?.parameters_schema ?? {},
              ),
      },
      controller: {
        id: training ? f.training_controller : f.controller,
        version: "1",
        parameters: schemaDefaults(
          plugins.find(
            (p) =>
              p.kind === "controller" &&
              p.id === (training ? f.training_controller : f.controller),
          )?.parameters_schema ?? {},
        ),
      },
      value_function: {
        id: training ? "zero" : f.value_function,
        version: "1",
        parameters: schemaDefaults(
          plugins.find(
            (p) =>
              p.kind === "value_function" &&
              p.id === (training ? "zero" : f.value_function),
          )?.parameters_schema ?? {},
        ),
      },
      solver: { ...spec.solver, backend: meta.backends[model] },
      evaluation: {
        ...spec.evaluation,
        accounting_version: meta.accounting,
        information_set: f.information_set ?? "online",
        metrics: meta.metrics.map((m: Json) => m.key),
      },
      dataset: { ...spec.dataset, split: training ? "train" : "test" },
      execution: {
        ...spec.execution,
        training_seeds: training ? [0] : undefined,
      },
    });
  };
  const framework =
    meta.frameworks.find((f: Json) => f.id === spec.framework_id) ??
    meta.frameworks.find(
      (f: Json) =>
        (f.model ?? spec.model.id) === spec.model.id &&
        ((f.controller === spec.controller.id &&
          f.value_function === spec.value_function.id) ||
          (f.training_controller === spec.controller.id &&
            spec.value_function.id === "zero")),
    ) ??
    meta.frameworks[0];
  const pluginSelect = (kind: string) => {
    const ref = spec[kind],
      plugin = plugins.find((p) => p.kind === kind && p.id === ref.id);
    return (
      <section className="card" key={kind}>
        <h3>
          {
            (
              {
                model: "模型参数",
                controller: "框架参数",
                value_function: "价值组件参数",
              } as Json
            )[kind]
          }
        </h3>
        <p>{ref.id}</p>
        {plugin && (
          <ParameterFields
            schema={plugin.parameters_schema}
            value={ref.parameters}
            onChange={(parameters) =>
              setSpec({ ...spec, [kind]: { ...ref, parameters } })
            }
          />
        )}
        {!Object.keys(plugin?.parameters_schema.properties ?? {}).length && (
          <small>当前组件无可调参数。</small>
        )}
      </section>
    );
  };
  const current = trace[scenario];
  const steps = current?.steps ?? [];
  const step = steps.find((s: Json) => s.period === period) ?? steps[0];
  const runSelector = (
    <select
      aria-label="选择运行"
      value={selected}
      onChange={(e) => {
        setSelected(e.target.value);
        setPeriod(0);
      }}
    >
      <option value="">选择运行…</option>
      {runs.map((r) => (
        <option value={r.run_id} key={r.run_id}>
          {r.spec.name} · {r.run_id}
        </option>
      ))}
    </select>
  );
  const createDataset = () =>
    act(async () => {
      const d = await api("/datasets/generate", {
        ...generation,
        family: meta.dataset_family,
      });
      await refresh();
      chooseData(d);
      return d;
    }, "冻结场景已生成，修订 hash 已选入编辑器");
  return (
    <div
      className="app"
      style={{ "--accent": meta.accent } as React.CSSProperties}
    >
      <aside>
        <div className="brand">
          <span className="brand-icon">◈</span>
          <div>
            DRL CO<small>EXPERIMENT WORKSPACE</small>
          </div>
        </div>
        <div className="nav-label">研究工作台</div>
        <nav>
          {pages
            .filter((p) => p !== "BHH 分析" || meta.id === "bhh")
            .map((p, i) => (
              <button
                key={p}
                className={page === p ? "active" : ""}
                onClick={() => {
                  setPage(p);
                  setResult(null);
                }}
              >
                <span>
                  {["◫", "▤", "⌘", "≡", "▧", "⇄", "▷", "⌁", "⤓", "⊙"][i]}
                </span>
                {p}
              </button>
            ))}
        </nav>
        <label className="model-selector">
          论文模型族
          <select
            aria-label="论文模型族"
            value={meta.id}
            onChange={(e) => onFamily(e.target.value)}
          >
            {families.map((f) => (
              <option key={f.id} value={f.id}>
                {f.name}
              </option>
            ))}
          </select>
          <small>参数 · 数据 · 框架 · 数学 · 结果完全切换</small>
        </label>
        <div className="sidebar-footer">
          <span className="dot" />
          本地工作空间<small>冻结数据 · 独立任务 · 可追溯</small>
        </div>
      </aside>
      <main>
        <header>
          <div>
            <div className="eyebrow">DRL CO / LOCAL RESEARCH PLATFORM</div>
            <h1>
              {page} <span className="family-name">{meta.name}</span>
            </h1>
          </div>
          <div className="badge">LOCAL ONLY · v0.2</div>
        </header>
        {error && (
          <div role="alert" className="error">
            {error}
          </div>
        )}
        {notice && (
          <div className="notice">
            {notice}
            <button onClick={() => setNotice("")}>×</button>
          </div>
        )}
        {page === "总览" && (
          <>
            <div className="hero">
              <div>
                <span className="eyebrow">REPRODUCIBLE BY DESIGN</span>
                <h2>把每一次实验，变成可验证的证据。</h2>
                <p>
                  {meta.name}{" "}
                  独立研究工作区。统一批实验协议，记录真实决策，清楚区分运营收益与求解目标。
                </p>
                <button
                  className="primary"
                  onClick={() => setPage("实验编辑器")}
                >
                  创建实验 →
                </button>
              </div>
              <div className="hero-art">
                ◈<small>DATA → POLICY → EVIDENCE</small>
              </div>
            </div>
            <div className="stats">
              {[
                ["冻结修订", datasets.length],
                ["已完成", runs.filter((r) => r.status === "COMPLETED").length],
                ["执行中", runs.filter((r) => r.status === "RUNNING").length],
                [
                  "需检查",
                  runs.filter((r) =>
                    ["FAILED", "TIMEOUT", "INTERRUPTED"].includes(r.status),
                  ).length,
                ],
              ].map(([label, n]) => (
                <div className="stat" key={label}>
                  <small>{label}</small>
                  <strong>{n}</strong>
                </div>
              ))}
            </div>
            <section className="card">
              <h3>
                最近运行{" "}
                <button onClick={() => setPage("队列")}>查看全部 →</button>
              </h3>
              <RunTable
                runs={runs.slice(0, 6)}
                open={(id) => {
                  setSelected(id);
                  setPage("运行详情");
                }}
              />
            </section>
            <div className="two">
              <section className="card">
                <h3>能力状态</h3>
                <p>
                  Gurobi 许可证：
                  {capabilities?.gurobi.license_verified ? "已验证" : "未验证"}
                </p>
                <p>CPU 运行；断点续算未开放；macOS 原生启动未验证。</p>
              </section>
              <section className="card">
                <h3>研究口径</h3>
                <p>监督学习价值拟合 ≠ SAC 强化学习。</p>
                <p>Oracle、在线策略和理论稳态不混合排名。</p>
              </section>
            </div>
          </>
        )}
        {page === "数据管理" && (
          <>
            <section className="card">
              <h3>生成可重复的示例数据</h3>
              <p>
                生成当前模型包的数据。建议用 N 个不同 seed 的 test
                场景形成统计样本；学习组件使用独立 train / validation 场景。
              </p>
              <ParameterFields
                schema={meta.generation_schema}
                value={generation}
                onChange={setGeneration}
              />
              <button className="primary" onClick={createDataset}>
                生成冻结修订
              </button>
            </section>
            <section className="card">
              <h3>数据修订</h3>
              <table>
                <thead>
                  <tr>
                    <th>数据集</th>
                    <th>版本 hash</th>
                    <th>物理模型</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((d) => (
                    <tr key={d.dataset_id + d.revision}>
                      <td>
                        {d.dataset_id}
                        {d.archived ? " · 已归档" : ""}
                      </td>
                      <td title={d.revision}>
                        <code>{d.revision.slice(0, 12)}</code>
                      </td>
                      <td>{d.family}</td>
                      <td>
                        <button
                          onClick={() =>
                            act(async () => {
                              const data = await api(
                                `/datasets/${d.dataset_id}/revisions/${d.revision}`,
                              );
                              setPreview(data);
                              return { scenarios: data.scenarios.length };
                            }, "预览已加载")
                          }
                        >
                          预览
                        </button>
                        <button
                          onClick={() => {
                            chooseData(d);
                            setPage("实验编辑器");
                          }}
                        >
                          选用
                        </button>
                        <button
                          onClick={() =>
                            act(() =>
                              api("/datasets/validate", {
                                dataset_id: d.dataset_id,
                                revision: d.revision,
                                split: "all",
                              }),
                            )
                          }
                        >
                          校验
                        </button>
                        <button
                          onClick={() =>
                            act(() =>
                              api("/datasets/" + d.dataset_id + "/archive", {}),
                            )
                          }
                        >
                          归档
                        </button>
                        <a
                          href={
                            "/api/v1/datasets/" +
                            d.dataset_id +
                            "/revisions/" +
                            d.revision
                          }
                          target="_blank"
                          rel="noreferrer"
                        >
                          导出 JSON
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
            {preview && (
              <section className="card">
                <h3>数据预览</h3>
                <JsonView
                  value={preview.scenarios.map((s: Json) => ({
                    ...s,
                    orders: s.orders.slice(0, 10),
                  }))}
                />
              </section>
            )}
            <section className="card">
              <h3>导入订单表</h3>
              <p>
                先预览一个同物理模型的数据集作为网络模板。表头需包含 id,
                departure, destination, passenger, book_time, start_time,
                end_time, revenue, penalty；自定义映射可通过 API 使用。
              </p>
              <select
                value={importFormat}
                onChange={(e) => setImportFormat(e.target.value)}
              >
                <option>csv</option>
                <option>json</option>
                <option>jsonl</option>
              </select>
              <input
                type="file"
                accept=".csv,.json,.jsonl"
                onChange={(e) =>
                  e.target.files?.[0]?.text().then(setImportText)
                }
              />
              <textarea
                value={importText}
                onChange={(e) => setImportText(e.target.value)}
                placeholder="粘贴订单数据"
              />
              <button
                disabled={!preview}
                onClick={() =>
                  act(async () => {
                    const template = {
                      ...preview!.scenarios[0],
                      family: preview!.family,
                    };
                    delete template.orders;
                    return api("/datasets/import", {
                      dataset_id: "import-" + Date.now(),
                      content: importText,
                      format: importFormat,
                      mapping: Object.fromEntries(
                        [
                          "id",
                          "departure",
                          "destination",
                          "passenger",
                          "book_time",
                          "start_time",
                          "end_time",
                          "revenue",
                          "penalty",
                        ].map((k) => [k, k]),
                      ),
                      template,
                    });
                  })
                }
              >
                校验并冻结导入
              </button>
            </section>
          </>
        )}
        {page === "实验编辑器" && (
          <>
            <section className="card">
              <h3>模型包实验模板</h3>
              <select
                aria-label="实验模板"
                defaultValue=""
                onChange={(e) => {
                  const preset = meta.templates.find(
                    (t: Json) => t.id === e.target.value,
                  )?.config;
                  if (!preset) return;
                  const draft = defaultSpec(meta);
                  for (const [key, value] of Object.entries(preset)) {
                    draft[key] =
                      typeof value === "object" &&
                      value !== null &&
                      !Array.isArray(value)
                        ? { ...draft[key], ...(value as Json) }
                        : value;
                  }
                  draft.family = meta.id;
                  draft.framework_id = undefined;
                  draft.dataset = {
                    ...spec.dataset,
                    split: draft.dataset.split ?? spec.dataset.split,
                    scenario_ids: [],
                  };
                  for (const kind of [
                    "model",
                    "controller",
                    "value_function",
                  ]) {
                    const p = plugins.find(
                      (p) => p.kind === kind && p.id === draft[kind].id,
                    );
                    draft[kind].parameters = {
                      ...schemaDefaults(p?.parameters_schema ?? {}),
                      ...draft[kind].parameters,
                    };
                  }
                  setSpec(draft);
                }}
              >
                <option value="">选择当前模型包的实验模板…</option>
                {meta.templates
                  .filter((t: Json) => t.config.model)
                  .map((t: Json) => (
                    <option key={t.id} value={t.id}>
                      {t.id}
                    </option>
                  ))}
              </select>
              <small>
                模板不会跨模型族；保留当前选择的冻结数据。学习模板仍需正确的数据
                split 与 checkpoint。
              </small>
            </section>

            <div className="toolbar">
              <input
                aria-label="实验名称"
                value={spec.name}
                onChange={(e) => setSpec({ ...spec, name: e.target.value })}
              />
              <select
                aria-label="数据集修订"
                value={spec.dataset.revision}
                onChange={(e) => {
                  const d = datasets.find((d) => d.revision === e.target.value);
                  if (d) chooseData(d);
                }}
              >
                <option value="">选择冻结数据…</option>
                {datasets
                  .filter((d) => !d.archived)
                  .map((d) => (
                    <option value={d.revision} key={d.dataset_id + d.revision}>
                      {d.dataset_id} · {d.revision.slice(0, 8)}
                    </option>
                  ))}
              </select>
              <select
                value={spec.dataset.split}
                onChange={(e) =>
                  setSpec({
                    ...spec,
                    dataset: { ...spec.dataset, split: e.target.value },
                  })
                }
              >
                {["train", "validation", "test", "all"].map((x) => (
                  <option key={x}>{x}</option>
                ))}
              </select>
            </div>
            <section className="card model-card">
              <h3>当前实验框架</h3>
              <select
                aria-label="实验框架"
                value={framework.id}
                onChange={(e) => {
                  const f = meta.frameworks.find(
                    (f: Json) => f.id === e.target.value,
                  );
                  selectFramework(f, !!f.training_required);
                }}
              >
                {meta.frameworks.map((f: Json) => (
                  <option key={f.id} value={f.id}>
                    {f.label}
                  </option>
                ))}
              </select>
              {framework.training_required && (
                <label>
                  学习组件阶段
                  <select
                    aria-label="学习组件阶段"
                    value={
                      spec.controller.id.startsWith("train_")
                        ? "train"
                        : "evaluation"
                    }
                    onChange={(e) =>
                      selectFramework(framework, e.target.value === "train")
                    }
                  >
                    <option value="train">Training / 拟合</option>
                    <option value="evaluation">
                      Held-out evaluation / 评估
                    </option>
                  </select>
                </label>
              )}
              <p>
                {framework.training_required
                  ? "该组件需要 checkpoint；拟合和评估使用不同的数据 split。"
                  : "当前为非学习型实验框架，直接运行，无拟合阶段。"}
              </p>
            </section>
            <MathPanel family={meta} spec={spec} />
            <div className="three">
              {["model", "controller", "value_function"].map(pluginSelect)}
            </div>
            <div className="two">
              <section className="card">
                <h3>求解资源</h3>
                <ParameterFields
                  schema={{
                    properties: {
                      time_limit_seconds: {
                        type: "number",
                        default: 30,
                        minimum: 0.1,
                      },
                      mip_gap: {
                        type: "number",
                        default: 0,
                        minimum: 0,
                        maximum: 1,
                      },
                      threads: {
                        type: "integer",
                        default: 1,
                        minimum: 1,
                        maximum: 16,
                      },
                      seed: { type: "integer", default: 0, minimum: 0 },
                    },
                  }}
                  value={spec.solver.parameters}
                  onChange={(parameters) =>
                    setSpec({ ...spec, solver: { ...spec.solver, parameters } })
                  }
                />
              </section>
              <section className="card">
                <h3>评估与终点</h3>
                <label>
                  信息集
                  <select
                    value={spec.evaluation.information_set}
                    onChange={(e) =>
                      setSpec({
                        ...spec,
                        evaluation: {
                          ...spec.evaluation,
                          information_set: e.target.value,
                        },
                      })
                    }
                  >
                    <option>online</option>
                    <option>oracle</option>
                  </select>
                </label>
                <label>
                  终点策略
                  <select
                    value={spec.evaluation.terminal_policy}
                    onChange={(e) =>
                      setSpec({
                        ...spec,
                        evaluation: {
                          ...spec.evaluation,
                          terminal_policy: e.target.value,
                        },
                      })
                    }
                  >
                    <option>report_pending</option>
                    <option>drain_committed</option>
                  </select>
                </label>
                <p className="muted">
                  不支持的组合由核心拒绝，不会自动换算法。BHH full-horizon
                  必须声明 oracle。
                </p>
              </section>
            </div>
            <BatchBuilder
              spec={spec}
              plugins={plugins}
              onApply={(b) => setRaw(pretty(b))}
            />
            <section className="card">
              <h3>完整 JSON / 批次扫描</h3>
              <textarea
                className="code-editor"
                value={raw}
                onChange={(e) => setRaw(e.target.value)}
              />
              <div className="row">
                <button
                  onClick={() =>
                    act(
                      () => api("/experiments/validate", JSON.parse(raw)),
                      "配置校验通过",
                    )
                  }
                >
                  校验配置
                </button>
                <button
                  onClick={() =>
                    act(
                      () => api("/experiments/plan", JSON.parse(raw)),
                      "计划已展开；未运行求解",
                    )
                  }
                >
                  预览计划
                </button>
                <button
                  className="primary"
                  onClick={() =>
                    act(async () => {
                      const batch = await api("/batches", JSON.parse(raw));
                      await refresh();
                      setPage("队列");
                      return batch;
                    }, "任务已加入队列")
                  }
                >
                  提交实验 →
                </button>
              </div>
            </section>
          </>
        )}
        {page === "队列" && (
          <section className="card">
            <h3>
              独立任务队列 <small>默认并发 1；单个求解器默认 1 线程</small>
            </h3>
            <RunTable
              runs={runs}
              open={(id) => {
                setSelected(id);
                setPage("运行详情");
              }}
              cancel={(id) =>
                act(() => api("/runs/" + id + "/cancel", {}), "已请求取消")
              }
            />
          </section>
        )}
        {page === "运行详情" && (
          <>
            {runSelector}
            {detail && (
              <>
                <div className="row spaced">
                  <h2>{detail.spec.name}</h2>
                  <span className={"status " + detail.status}>
                    {detail.status}
                  </span>
                  <button
                    onClick={() =>
                      act(
                        () => api("/runs/" + selected + "/rerun", {}),
                        "同配置已创建新运行",
                      )
                    }
                  >
                    重跑
                  </button>
                  <button
                    onClick={() => {
                      setSpec(detail.spec);
                      setPage("实验编辑器");
                    }}
                  >
                    复制配置
                  </button>
                </div>
                <div className="two">
                  <section className="card">
                    <h3>场景指标</h3>
                    <JsonView
                      value={detail.metrics ?? { reason: "运行尚无完整结果" }}
                    />
                  </section>
                  <section className="card">
                    <h3>时间序列 · 指派时利润</h3>
                    {steps.length ? (
                      <ResponsiveContainer width="100%" height={250}>
                        <LineChart
                          data={steps.map((s: Json) => ({
                            period: s.period,
                            profit: s.after?.profit,
                          }))}
                        >
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="period" />
                          <YAxis />
                          <Tooltip />
                          <Line
                            type="monotone"
                            dataKey="profit"
                            stroke="#138579"
                            dot={false}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <p>没有适用的利润轨迹。</p>
                    )}
                  </section>
                </div>
                <section className="card">
                  <h3>事件与产物</h3>
                  <div className="artifacts">
                    {detail.artifacts.map((a: string) => (
                      <a
                        key={a}
                        href={`/api/v1/runs/${selected}/artifacts/${a}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {a} ↗
                      </a>
                    ))}
                  </div>
                  <JsonView value={detail.events.slice(-12)} />
                </section>
                <details className="card">
                  <summary>精确配置与数据引用</summary>
                  <JsonView value={detail.spec} />
                </details>
              </>
            )}
          </>
        )}
        {page === "实验比较" && (
          <>
            <section className="card">
              <h3>同场景配对比较</h3>
              <p>
                必须使用相同
                hash、物理参数、信息集与记账策略。失败任务不被静默排除。一个场景不计算置信区间。
              </p>
              {runs.map((r) => (
                <label className="check" key={r.run_id}>
                  <input
                    type="checkbox"
                    checked={comparisonIds.includes(r.run_id)}
                    onChange={(e) =>
                      setComparisonIds(
                        e.target.checked
                          ? [...comparisonIds, r.run_id]
                          : comparisonIds.filter((id) => id !== r.run_id),
                      )
                    }
                  />
                  {r.spec.name} · {r.run_id} · {r.status}
                </label>
              ))}
              <button
                disabled={comparisonIds.length !== 2}
                className="primary"
                onClick={() =>
                  act(() =>
                    api("/comparisons", {
                      run_ids: comparisonIds,
                      metric: runs
                        .find((r) => r.run_id === comparisonIds[0])
                        ?.spec.model.id.startsWith("bhh")
                        ? "business_cost"
                        : "operating_profit",
                    }),
                  )
                }
              >
                比较 A − B
              </button>
            </section>
            {result?.pairs && (
              <section className="card">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={result.pairs}>
                    <XAxis dataKey="scenario_id" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="difference" fill="#138579" />
                  </BarChart>
                </ResponsiveContainer>
              </section>
            )}
          </>
        )}
        {page === "回放" && (
          <>
            {runSelector}
            {current ? (
              <>
                <div className="toolbar">
                  <select
                    value={scenario}
                    onChange={(e) => {
                      setScenario(Number(e.target.value));
                      setPeriod(0);
                    }}
                  >
                    {trace.map((s, i) => (
                      <option key={s.scenario_id} value={i}>
                        {s.scenario_id}
                      </option>
                    ))}
                  </select>
                  <label>
                    期数 {period}
                    <input
                      type="range"
                      min={0}
                      max={Math.max(...steps.map((s: Json) => s.period), 0)}
                      value={period}
                      onChange={(e) => setPeriod(Number(e.target.value))}
                    />
                  </label>
                </div>
                <div className="two">
                  <section className="card">
                    <h3>
                      {step?.committed
                        ? "实际已提交状态"
                        : "求解计划 · 尚未提交"}
                    </h3>
                    <Network network={current.network} state={step?.after} />
                  </section>
                  <section className="card">
                    <h3>决策与资源事件</h3>
                    <JsonView value={step} />
                  </section>
                </div>
              </>
            ) : (
              <section className="empty">
                选择带轨迹的运行。BHH 计划不绘制成已经发生的车辆移动。
              </section>
            )}
          </>
        )}
        {page === "BHH 分析" && (
          <>
            <StudyHeatmap runs={runs} />
            <section className="card">
              <h3>连续理论与有限调度分开阅读</h3>
              <p>
                稳态：两城市对称、可分割货物、无限车队。有限模型：整数共享
                HV/AV、离散时段、时间窗与三段货物流。两个 cost
                不能直接混为同一口径。
              </p>
              <div className="row">
                <button
                  onClick={() => {
                    setPage("数据管理");
                  }}
                >
                  准备 BHH 数据
                </button>
                <button
                  onClick={() => {
                    selectFramework(
                      meta.frameworks.find((f: Json) => f.id === "steady"),
                    );
                    setPage("实验编辑器");
                  }}
                >
                  配置稳态分析
                </button>
                <button
                  onClick={() => {
                    selectFramework(
                      meta.frameworks.find((f: Json) => f.id === "rolling"),
                    );
                    setPage("实验编辑器");
                  }}
                >
                  配置真实滚动窗口
                </button>
              </div>
            </section>
            {runSelector}
            {detail?.spec.model.id.startsWith("bhh") && (
              <section className="card">
                <h3>分解与窗口结果</h3>
                <JsonView value={detail.metrics} />
                <JsonView value={trace} />
              </section>
            )}
          </>
        )}
        {page === "复现" && (
          <>
            <section className="card">
              <h3>复现不是一个 hash</h3>
              <p>
                导出包含精确数据、配置、schema、环境、依赖锁和 dirty
                源码快照（若有）。导入检查只验证内容，不自动执行不可信代码或反序列化
                checkpoint。
              </p>
              {runSelector}
              <button
                disabled={!selected}
                onClick={() =>
                  act(
                    () => api("/reproduction/export", { run_id: selected }),
                    "复现包已生成；请从运行产物下载",
                  )
                }
              >
                导出复现包
              </button>
              {selected && (
                <a href={`/api/v1/runs/${selected}/artifacts/reproduction.zip`}>
                  下载复现包
                </a>
              )}
              <label>
                检查复现包
                <input
                  type="file"
                  accept=".zip"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file)
                      act(async () => {
                        if (!token) token = (await api("/session")).write_token;
                        const r = await fetch("/api/v1/reproduction/validate", {
                          method: "POST",
                          headers: { "X-Workspace-Token": token },
                          body: await file.arrayBuffer(),
                        });
                        const data = await r.json();
                        if (!r.ok) throw Error(pretty(data));
                        return data;
                      });
                  }}
                />
              </label>
            </section>
          </>
        )}
        {page === "数学模型" && <MathPanel family={meta} spec={spec} />}
        {page === "实验分析" && (
          <AnalysisPanel family={meta} runs={runs} api={api} />
        )}
        {page === "诊断" && (
          <>
            <section className="card">
              <h3>本地环境</h3>
              <JsonView value={capabilities} />
            </section>
            <section className="card">
              <h3>任务诊断与只求解单步</h3>
              {runSelector}
              <div className="row">
                <button
                  disabled={!selected}
                  onClick={() =>
                    act(() => api("/diagnostics", { run_id: selected }))
                  }
                >
                  生成脱敏诊断
                </button>
                <input
                  aria-label="调试期数"
                  type="number"
                  value={period}
                  min={0}
                  onChange={(e) => setPeriod(Number(e.target.value))}
                />
                <button
                  onClick={() =>
                    act(
                      () => api("/debug/step", { spec, period }),
                      "单步已求解，未提交该期决策",
                    )
                  }
                >
                  当前配置单步
                </button>
              </div>
              <p>
                终端：scripts/exp.ps1 doctor --json；debug step --config …
                --period …；网页不提供任意命令执行。
              </p>
            </section>
          </>
        )}
        {result && (
          <details className="card" open>
            <summary>操作结果</summary>
            <JsonView value={result} />
          </details>
        )}
        <footer>DRL CO · 每个数字都有数据、配置和执行记录。</footer>
      </main>
    </div>
  );
}
function RunTable({
  runs,
  open,
  cancel,
}: {
  runs: Json[];
  open: (id: string) => void;
  cancel?: (id: string) => void;
}) {
  return runs.length ? (
    <table>
      <thead>
        <tr>
          <th>实验</th>
          <th>模型 / 控制器</th>
          <th>状态</th>
          <th>进度</th>
          <th>操作</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((r) => (
          <tr key={r.run_id}>
            <td>
              <button className="link" onClick={() => open(r.run_id)}>
                {r.spec.name}
              </button>
              <small>{r.run_id}</small>
            </td>
            <td>
              {r.spec.model.id}
              <small>{r.spec.controller.id}</small>
            </td>
            <td>
              <span className={"status " + r.status}>{r.status}</span>
            </td>
            <td>
              {Math.round(
                (r.events.filter((e: Json) => e.progress !== undefined).at(-1)
                  ?.progress ?? 0) * 100,
              )}
              %
            </td>
            <td>
              <button onClick={() => open(r.run_id)}>打开 →</button>
              {cancel && ["QUEUED", "RUNNING"].includes(r.status) && (
                <button onClick={() => cancel(r.run_id)}>取消</button>
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  ) : (
    <div className="empty">还没有运行。先准备数据，再提交一个实验。</div>
  );
}
function App() {
  const [families, setFamilies] = useState<Json[]>([]),
    [id, setId] = useState(localStorage.getItem("model-family") ?? "single"),
    [error, setError] = useState("");
  useEffect(() => {
    requestApi("/families")
      .then(setFamilies)
      .catch((e) => setError(String(e)));
  }, []);
  const meta = families.find((f) => f.id === id) ?? families[0];
  if (!meta)
    return (
      <main>
        <h1>实验平台</h1>
        <p>{error || "加载独立模型包…"}</p>
      </main>
    );
  return (
    <Workspace
      key={meta.id}
      meta={meta}
      families={families}
      onFamily={(id) => {
        localStorage.setItem("model-family", id);
        setId(id);
      }}
    />
  );
}
createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
