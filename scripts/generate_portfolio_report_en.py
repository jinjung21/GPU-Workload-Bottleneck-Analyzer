#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Frame, PageBreak, PageTemplate, Paragraph, Spacer

from generate_portfolio_report import (
    FINAL_OUTPUT,
    GOLD,
    INK,
    LINE,
    MUTED,
    PALE,
    ROOT,
    TEAL,
    WHITE,
    _escape_code,
    _image,
    _metric_cards,
    _rule,
    _size_sweep_decision_rows,
    _table,
    _two_column_table,
)


OUTPUT = ROOT / "output" / "pdf" / "gpu_pim_bottleneck_analyzer_portfolio_report_en.pdf"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = EnglishPortfolioDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=17 * mm,
        leftMargin=17 * mm,
        topMargin=19 * mm,
        bottomMargin=18 * mm,
        title="GPU-PIM Workload Bottleneck Analyzer",
        author="GPU Workload Bottleneck Analyzer Project",
        subject="English portfolio technical report for GPU, memory systems, and PIM/NMP analysis",
    )
    doc.build(_story(styles))
    print(f"Wrote English portfolio report: {OUTPUT}")


class EnglishPortfolioDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs: object) -> None:
        super().__init__(filename, **kwargs)
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="body")
        self.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=self._draw_page))

    def _draw_page(self, canvas: object, doc: object) -> None:
        page = canvas.getPageNumber()
        if page == 1:
            canvas.setFillColor(TEAL)
            canvas.rect(0, A4[1] - 11 * mm, A4[0], 11 * mm, fill=1, stroke=0)
            return
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.line(self.leftMargin, A4[1] - 13 * mm, A4[0] - self.rightMargin, A4[1] - 13 * mm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, A4[1] - 10 * mm, "GPU-PIM Workload Bottleneck Analyzer")
        canvas.drawRightString(A4[0] - self.rightMargin, 9 * mm, f"{page}")
        canvas.restoreState()


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "EnglishTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=32,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "EnglishSubtitle",
            fontName="Helvetica",
            fontSize=12.5,
            leading=18,
            textColor=TEAL,
            spaceAfter=8,
        ),
        "h1": ParagraphStyle(
            "EnglishH1",
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=24,
            textColor=INK,
            spaceBefore=3,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "EnglishH2",
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=17,
            textColor=TEAL,
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "EnglishBody",
            fontName="Helvetica",
            fontSize=9.3,
            leading=14.5,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "EnglishSmall",
            fontName="Helvetica",
            fontSize=7.6,
            leading=11,
            textColor=MUTED,
        ),
        "caption": ParagraphStyle(
            "EnglishCaption",
            fontName="Helvetica",
            fontSize=7.5,
            leading=11,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "callout": ParagraphStyle(
            "EnglishCallout",
            fontName="Helvetica",
            fontSize=9.2,
            leading=14.5,
            textColor=INK,
            leftIndent=8,
            rightIndent=8,
            borderColor=TEAL,
            borderWidth=1,
            borderPadding=8,
            backColor=PALE,
            spaceBefore=6,
            spaceAfter=8,
        ),
        "code": ParagraphStyle(
            "EnglishCode",
            fontName="Courier",
            fontSize=6.8,
            leading=9.7,
            textColor=INK,
            leftIndent=7,
            rightIndent=7,
            borderColor=LINE,
            borderWidth=0.6,
            borderPadding=7,
            backColor=colors.HexColor("#F7F9FA"),
            spaceBefore=4,
            spaceAfter=7,
        ),
    }


def _story(s: dict[str, ParagraphStyle]) -> list[object]:
    story: list[object] = []
    story += _cover(s)
    story += _executive_summary(s)
    story += _problem(s)
    story += _architecture(s)
    story += _method(s)
    story += _evolution(s)
    story += _setup(s)
    story += _roofline(s)
    story += _ncu(s)
    story += _simulator(s)
    story += _size_sweep(s)
    story += _validation(s)
    story += _reproduction(s)
    story += _conclusion(s)
    return story


def _cover(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Spacer(1, 31 * mm),
        Paragraph("GPU-PIM Workload<br/>Bottleneck Analyzer", s["title"]),
        Paragraph(
            "A reproducible GPU, memory-system, and PIM/NMP architecture exploration project",
            s["subtitle"],
        ),
        Spacer(1, 8 * mm),
        _rule(TEAL, 150 * mm),
        Spacer(1, 9 * mm),
        Paragraph(
            "<b>Portfolio Technical Report</b><br/>"
            "Roofline analysis, CUDA microbenchmarks, Nsight Compute counters, SAIT PIMSimulator, "
            "and problem-size sweeps integrated into one traceable workflow.",
            s["body"],
        ),
        Spacer(1, 22 * mm),
        _metric_cards(
            [
                ("9", "CUDA workloads"),
                ("v6", "cache/stall-aware model"),
                ("31", "automated tests"),
                ("2.36x", "modeled policy speedup"),
            ],
            s,
        ),
        Spacer(1, 28 * mm),
        Paragraph(
            f"Final English edition: {date.today().isoformat()}<br/>"
            "Target: NVIDIA GeForce RTX 2080 Ti | CUDA 11.0 | Nsight Compute 2020.1.1",
            s["small"],
        ),
        PageBreak(),
    ]


def _executive_summary(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("Executive Summary", s["h1"]),
        Paragraph(
            "This project goes beyond classifying a GPU kernel as memory-bound or compute-bound. "
            "It asks whether the kernel is a reasonable PIM/NMP exploration candidate and records the evidence behind that decision. "
            "CUDA events provide measured GPU runtime, the Roofline model establishes a first-order bound, and Nsight Compute counters refine the diagnosis using DRAM, cache, scheduler, and dependency-stall behavior.",
            s["body"],
        ),
        Paragraph(
            "<b>Final outcome.</b> Seven workloads remain PIM/NMP exploration candidates: vector addition, SAXPY, random gather, reduction, scan, matrix transpose, and GEMV. "
            "The high-reuse tiled GEMM and optimized cuBLAS SGEMM remain on the GPU. The v6 policy separates all seven positive labels from two negative controls on the calibration set. "
            "This is calibration alignment, not held-out generalization accuracy.",
            s["callout"],
        ),
        Paragraph("Key results", s["h2"]),
        _table(
            [
                ["Item", "Result", "Interpretation"],
                ["Measured GPU-only runtime", "10.963 ms", "Sum of nine CUDA-event measurements"],
                ["Final v6 policy estimate", "4.638 ms / 2.36x", "Four simulator-backed and three analytical offloads"],
                ["Simulator coverage", "4/9 (44.4%)", "vector_add, saxpy, reduction, and gemv"],
                ["Calibration controls", "3/3 pass", "GEMV positive; tiled and cuBLAS GEMM negative"],
                ["Size stability", "Three scales", "The 7 PIM/NMP to 2 GPU split remains stable"],
            ],
            [43 * mm, 43 * mm, 86 * mm],
            s,
        ),
        Spacer(1, 5 * mm),
        Paragraph("Report map", s["h2"]),
        Paragraph(
            "1. Problem and background | 2. System architecture | 3. Method | 4. Model evolution | "
            "5. Experimental setup | 6. Roofline results | 7. NCU and v6 results | 8. Simulator normalization | "
            "9. Size sweep | 10. Validation and limitations | 11. Reproduction | 12. Conclusion",
            s["body"],
        ),
        PageBreak(),
    ]


def _problem(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("1. Problem Definition and Background", s["h1"]),
        Paragraph("1.1 Why GPU bottleneck classification matters", s["h2"]),
        Paragraph(
            "GPUs hide memory latency by scheduling many warps and provide high arithmetic throughput through many parallel execution units. "
            "A slow kernel, however, is not necessarily slow for the same reason as another kernel. Low arithmetic intensity can make DRAM bandwidth the limit. "
            "High arithmetic intensity can make compute throughput the limit. A kernel that uses neither resource effectively may instead be constrained by irregular access, cache misses, dependencies, divergence, low occupancy, or synchronization.",
            s["body"],
        ),
        Paragraph("1.2 What PIM/NMP attempts to change", s["h2"]),
        Paragraph(
            "Processing-in-Memory and Near-Memory Processing reduce data movement by performing selected operations close to memory. "
            "The strongest candidates therefore tend to have low arithmetic intensity, large memory traffic, limited cache reuse, or high memory-latency stalls. "
            "Dense matrix multiplication is often a poor candidate because GPUs exploit its reuse and compute density effectively.",
            s["body"],
        ),
        _two_column_table(
            "Signals favoring the GPU",
            ["High arithmetic intensity", "High L2 or shared-memory reuse", "Compute-heavy operations", "Optimized library support"],
            "Signals favoring PIM/NMP exploration",
            ["Low arithmetic intensity", "DRAM pressure or memory stalls", "Streaming or irregular access", "Low reuse and high data movement"],
            s,
        ),
        Paragraph("1.3 Project questions", s["h2"]),
        Paragraph(
            "Can measurable features separate GPU bottleneck classes? Can a cache-, stall-, reuse-, and transfer-aware model improve on low-AI rules? "
            "Can GPU measurements and PIM simulation be combined without mixing unrelated clocks? Are decisions stable across multiple problem sizes?",
            s["callout"],
        ),
        PageBreak(),
    ]


def _architecture(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("2. System Architecture", s["h1"]),
        Paragraph(
            "The repository separates data collection, feature extraction, model comparison, simulator attachment, policy evaluation, and reporting. "
            "Existing CSV files can be analyzed on a machine without CUDA, while the GPU server is responsible only for generating fresh measurements.",
            s["body"],
        ),
        _pipeline_drawing_en(),
        Paragraph("Figure 1. Data flow from GPU measurement to the final evidence-aware decision", s["caption"]),
        Paragraph("2.1 Input layer", s["h2"]),
        _table(
            [
                ["Input", "Role", "Current basis"],
                ["gpu_profile.csv", "Runtime, FLOPs, theoretical DRAM bytes", "RTX 2080 Ti CUDA events"],
                ["ncu_metrics.csv", "DRAM, cache, scheduler, and stall counters", "Nsight Compute 2020.1.1"],
                ["gpu_benchmark_metadata.csv", "Complexity, reuse, transfer, and control roles", "Calibration metadata"],
                ["sait_pim_simulation.csv", "PIM-on/PIM-off cycles and speedup", "SAIT PIMSimulator logs"],
            ],
            [45 * mm, 68 * mm, 59 * mm],
            s,
        ),
        Paragraph("2.2 Output layer", s["h2"]),
        Paragraph(
            "Each full run produces a Roofline figure, model-comparison figure, end-to-end policy figure, and Markdown report. "
            "The size sweep stores three raw profiles, three detailed reports, a combined CSV and Markdown summary, and a trend figure. "
            "Intermediate evidence is retained so the final result can be traced to its source.",
            s["body"],
        ),
        PageBreak(),
    ]


def _method(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("3. Analysis Method", s["h1"]),
        Paragraph("3.1 Roofline first-stage classification", s["h2"]),
        Paragraph(
            "Arithmetic intensity is the amount of floating-point work performed per byte transferred from DRAM. "
            "The RTX 2080 Ti configuration uses 13.45 TFLOP/s peak FP32 throughput and 616 GB/s peak memory bandwidth, producing a ridge point of 21.8344 FLOP/Byte.",
            s["body"],
        ),
        Paragraph(
            "AI = FLOPs / DRAM bytes<br/>"
            "Attainable performance = min(peak FLOP/s, peak bandwidth x AI)<br/>"
            "Roofline utilization = achieved performance / attainable performance",
            s["callout"],
        ),
        Paragraph("3.2 Cache/stall-aware feature_cost_v6", s["h2"]),
        Paragraph(
            "v6 extends the reuse-aware v4 model and the basic NCU-aware v5 model. It optionally uses cache hit rates, load/store efficiency, warp and branch efficiency, "
            "long and short scoreboard stalls, barrier stalls, eligible warp supply, register pressure, and transaction proxies. "
            "The available optional-counter ratio controls how far v6 may move from the v5 estimate.",
            s["body"],
        ),
        _table(
            [
                ["Signal group", "PIM opportunity", "GPU-resident signal"],
                ["Cache", "Low L1/L2 hit rate", "High locality or reuse"],
                ["Memory", "High DRAM pressure or dependency stalls", "Efficient coalescing and locality"],
                ["Scheduler", "Low eligible-warp supply", "GPU can hide latency"],
                ["Control", "Low divergence and barrier risk", "Complex control or synchronization"],
                ["Metadata", "Low complexity and high partitionability", "Compute, transfer, or reuse risk"],
            ],
            [35 * mm, 68 * mm, 69 * mm],
            s,
        ),
        Paragraph("3.3 Common-cost policy evaluation", s["h2"]),
        Paragraph(
            "AI-only, traffic-only, heuristic, analytical, and feature-cost models choose different workloads to offload. "
            "Their total runtime is evaluated with the same current v6 PIM timing table. This isolates decision quality from differences between timing formulas.",
            s["body"],
        ),
        PageBreak(),
    ]


def _evolution(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("4. Implementation and Model Evolution", s["h1"]),
        Paragraph(
            "The project evolved from a basic Roofline analyzer into an evidence-aware GPU/PIM research pipeline. "
            "Each model version addressed a concrete false positive, missing measurement, or comparability problem found in the preceding version.",
            s["body"],
        ),
        _table(
            [
                ["Stage", "Implementation", "Problem addressed"],
                ["v1-v2", "Roofline, heuristic score, and PrIM-inspired proxy", "Connect bottlenecks to literature categories"],
                ["v3", "GPU/PIM memory, compute, transfer, and sync cost", "Reduce low-AI false positives"],
                ["v4", "Data-reuse gate and cuBLAS SGEMM control", "Avoid misclassifying reusable GEMM"],
                ["v5", "Nsight Compute DRAM/SM/cache pressure", "Calibrate metadata proxies with counters"],
                ["v6", "Cache hit, scheduler, stalls, and coverage", "Separate bandwidth, locality, and latency causes"],
                ["Final", "Simulator ratio normalization and evidence tiers", "Stop directly adding unrelated clock domains"],
            ],
            [22 * mm, 72 * mm, 78 * mm],
            s,
        ),
        Spacer(1, 6 * mm),
        Paragraph("Key engineering decisions", s["h2"]),
        Paragraph(
            "- Store CUDA-event runtime separately from theoretical FLOP and byte counts.<br/>"
            "- Integrate external PIM tools through a small adapter CSV instead of vendoring a simulator.<br/>"
            "- Preserve missing NCU values as NA for compatibility with older profiler versions.<br/>"
            "- Parameterize benchmark sizes and iteration counts for repeatable experiments.<br/>"
            "- Protect parsers, models, simulator normalization, and policy contracts with 31 automated tests.",
            s["callout"],
        ),
        Paragraph("Code organization", s["h2"]),
        _table(
            [
                ["Area", "Primary files"],
                ["Analysis", "src/roofline.py, classifier.py, features.py"],
                ["Models", "src/cost_model_v3.py through cost_model_v6.py"],
                ["Integration", "src/model_comparison.py, simulator.py, end_to_end.py"],
                ["Measurement", "benchmarks/*.cu and scripts/profile_*.sh"],
                ["Reporting", "src/plot.py, src/report.py, and main.py"],
            ],
            [45 * mm, 127 * mm],
            s,
        ),
        PageBreak(),
    ]


def _setup(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("5. Experimental Setup and Workloads", s["h1"]),
        _two_column_table(
            "GPU server",
            ["Ubuntu 20.04", "GeForce RTX 2080 Ti", "Driver 535.230.02", "CUDA toolkit 11.0", "Host compiler g++-9"],
            "Profiling stack",
            ["CUDA event runtime", "nvprof raw logs", "Nsight Compute 2020.1.1", "SAIT PIMSimulator", "Python, pandas, and matplotlib"],
            s,
        ),
        Paragraph("5.1 Benchmark suite", s["h2"]),
        _table(
            [
                ["Workload", "Purpose", "Final role"],
                ["vector_add / saxpy", "Streaming bandwidth", "PIM/NMP candidates"],
                ["random_gather", "Irregular memory latency", "PIM/NMP candidate"],
                ["reduction / scan", "Collective memory primitives", "Candidates with synchronization risk"],
                ["matrix_transpose", "Memory-dominated layout conversion", "PIM/NMP candidate"],
                ["gemv", "Low-reuse matrix-vector multiply", "PIM-positive control"],
                ["matrix_mul_tiled", "Reuse-aware custom GEMM", "GPU negative control"],
                ["cublas_sgemm", "Optimized dense compute", "GPU negative control"],
            ],
            [48 * mm, 70 * mm, 54 * mm],
            s,
        ),
        Paragraph("5.2 Problem-size sweep", s["h2"]),
        Paragraph(
            "Vector inputs use 1,048,576, 4,194,304, and 16,777,216 elements. Matrix inputs use dimensions 512, 1024, and 1536. "
            "The sweep compares runtime, achieved bandwidth, bottleneck class, offload decision, and analytical speedup trend.",
            s["body"],
        ),
        Paragraph("5.3 Interpretation rule", s["h2"]),
        Paragraph(
            "Measured refers only to GPU runtime and profiler counters. Simulated refers to PIMSimulator cycles and ratios. "
            "Estimated refers to feature-cost and end-to-end policy outputs. The report maintains this vocabulary to avoid presenting model output as physical PIM measurement.",
            s["callout"],
        ),
        PageBreak(),
    ]


def _roofline(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("6. Roofline and GPU Measurement Results", s["h1"]),
        _image(FINAL_OUTPUT / "figures" / "roofline.png", 171 * mm, 112 * mm),
        Paragraph("Figure 2. Roofline positions of the nine workloads measured on the RTX 2080 Ti", s["caption"]),
        _table(
            [
                ["Group", "Representative result", "Diagnosis"],
                ["Streaming", "vector_add/saxpy near 555 GB/s and 90% Roofline", "DRAM bandwidth saturation"],
                ["Irregular", "random_gather 38.8 GB/s and 6.3% Roofline", "Latency/locality rather than bandwidth"],
                ["Collective", "reduction 40%; scan 27% Roofline", "Mixed memory and synchronization effects"],
                ["Dense", "cuBLAS SGEMM AI 341.3 and 10.85 TFLOP/s", "Strong compute-bound control"],
                ["Reuse", "tiled GEMM AI 3.97 and L2 hit 98.34%", "Low AI alone must not imply PIM"],
            ],
            [34 * mm, 79 * mm, 59 * mm],
            s,
        ),
        Paragraph(
            "Roofline alone places tiled GEMM in the memory region, yet its high L2 reuse makes it a poor PIM candidate. "
            "Random gather does not saturate DRAM but suffers extreme long-scoreboard stalls, making memory-latency reduction relevant. "
            "These two cases motivate an NCU-aware model.",
            s["callout"],
        ),
        PageBreak(),
    ]


def _ncu(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("7. Nsight Compute and v6 Results", s["h1"]),
        _image(FINAL_OUTPUT / "figures" / "model_comparison.png", 171 * mm, 74 * mm),
        Paragraph("Figure 3. Calibration-set alignment and the latest v6 analytical speedup estimates", s["caption"]),
        _table(
            [
                ["Kernel", "Key NCU evidence", "v6 interpretation"],
                ["vector_add", "SOL DRAM 85.41%; long scoreboard 91.20%", "Streaming PIM opportunity"],
                ["random_gather", "L2 hit 9.99%; long scoreboard 98.30%", "Irregular latency opportunity"],
                ["gemv", "L1 hit 87.78%; long scoreboard 67.50%", "Some locality, but simulator-positive"],
                ["matrix_mul_tiled", "L2 hit 98.34%", "Strong GPU locality advantage"],
                ["cublas_sgemm", "L2 hit 87.47%; compute-bound", "Remain on GPU"],
            ],
            [38 * mm, 76 * mm, 58 * mm],
            s,
        ),
        Paragraph("7.1 Final offload decisions", s["h2"]),
        _table(
            [
                ["PIM/NMP exploration candidates", "Remain on GPU"],
                ["vector_add, saxpy, random_gather", "matrix_mul_tiled"],
                ["reduction, scan, matrix_transpose, gemv", "cublas_sgemm"],
            ],
            [104 * mm, 68 * mm],
            s,
        ),
        Paragraph(
            "The installed NCU version provides four of twelve optional v6 counters per kernel, or 33% coverage. "
            "v6 uses the cache and stall evidence but limits its correction from v5 by that coverage ratio. Missing values are not treated as measured zeros.",
            s["callout"],
        ),
        PageBreak(),
    ]


def _simulator(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("8. PIM Simulator Integration and Time Normalization", s["h1"]),
        Paragraph("8.1 The comparability problem", s["h2"]),
        Paragraph(
            "The original integration converted SAIT PIMSimulator cycles using a temporary 1 ns/cycle value and directly combined that time with RTX 2080 Ti milliseconds. "
            "The simulator baseline and GPU are not the same architecture or clock domain. The conversion is useful for traceability but not for direct cross-platform addition.",
            s["body"],
        ),
        Paragraph(
            "raw_sim_time_ms = PIM cycles x cycle_time_ns / 10^6<br/>"
            "scaled_PIM_time_ms = measured_GPU_time_ms / simulator_speedup",
            s["callout"],
        ),
        Paragraph(
            "The final implementation preserves raw cycles and raw converted time as evidence. End-to-end totals use the simulator's internal PIM-on versus PIM-off speedup ratio applied to measured GPU runtime. "
            "Unmapped selected kernels use the v6 analytical estimate, and the report counts those fallbacks explicitly.",
            s["body"],
        ),
        _image(FINAL_OUTPUT / "figures" / "end_to_end.png", 171 * mm, 74 * mm),
        Paragraph("Figure 4. Offload policies evaluated with a common PIM timing table", s["caption"]),
        _table(
            [
                ["Metric", "Final value", "Evidence"],
                ["GPU-only", "10.963 ms", "Measured CUDA-event sum"],
                ["feature_cost_v6 policy", "4.638 ms", "Four simulator-scaled and three analytical offloads"],
                ["Modeled speedup", "2.36x", "Not measured PIM silicon speedup"],
                ["Simulator coverage", "44.4%", "Four of nine mapped workloads"],
            ],
            [43 * mm, 47 * mm, 82 * mm],
            s,
        ),
        PageBreak(),
    ]


def _size_sweep(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("9. Problem-Size Sweep", s["h1"]),
        _image(ROOT / "outputs" / "size_sweep" / "size_sweep.png", 178 * mm, 94 * mm),
        Paragraph("Figure 5. Measured GPU runtime and analytical PIM opportunity at three input sizes", s["caption"]),
        Paragraph(
            "Runtime increases normally with input size, with random gather showing the steepest growth. Streaming kernels preserve analytical opportunity at large sizes. "
            "Matrix transpose remains favorable, while tiled GEMM and cuBLAS SGEMM remain below 1x and stay on the GPU.",
            s["body"],
        ),
        _table(_size_sweep_decision_rows(), [46 * mm, 42 * mm, 42 * mm, 42 * mm], s),
        Paragraph(
            "The 7 PIM/NMP to 2 GPU decision split remains unchanged at all three sizes. This shows internal size stability within the current synthetic suite. "
            "It does not replace independent validation on unseen application workloads.",
            s["callout"],
        ),
        PageBreak(),
    ]


def _validation(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("10. Validation, Evidence, and Limitations", s["h1"]),
        Paragraph("10.1 Validation scope", s["h2"]),
        _table(
            [
                ["Validation", "Result", "Residual risk"],
                ["Automated tests", "31 passed", "GPU and NCU execution still requires the server"],
                ["Calibration labels", "v4/v6 F1 = 1.00", "Not held-out accuracy"],
                ["Control workloads", "3/3 pass", "Small control set"],
                ["Size sweep", "Three scales stable", "Limited application diversity"],
                ["PIM simulator", "4/9 mapped", "Mapping fidelity and hardware differences"],
            ],
            [42 * mm, 48 * mm, 82 * mm],
            s,
        ),
        Paragraph("10.2 Evidence tiers", s["h2"]),
        _table(
            [
                ["Tier", "Evidence", "Current workloads"],
                ["A", "Measured GPU + NCU + mapped PIM simulator", "vector_add, saxpy, reduction, gemv"],
                ["B", "Measured GPU + NCU + analytical PIM", "Remaining five final workloads"],
                ["C", "Measured GPU + analytical PIM", "Size-sweep runs without NCU"],
            ],
            [20 * mm, 83 * mm, 69 * mm],
            s,
        ),
        Paragraph("10.3 Limitations that must remain explicit", s["h2"]),
        Paragraph(
            "- No runtime was measured on physical PIM silicon.<br/>"
            "- Nine microbenchmark-style workloads cannot establish generalization.<br/>"
            "- FLOP and byte counts are theoretical implementation-level counts.<br/>"
            "- Nsight Compute 2020.1.1 provides partial optional-counter coverage.<br/>"
            "- Analytical bandwidth, throughput, transfer, and synchronization parameters require calibration.<br/>"
            "- The end-to-end model does not fully include placement, orchestration, programming, or deployment overhead.",
            s["callout"],
        ),
        Paragraph("10.4 Highest-value next research step", s["h2"]),
        Paragraph(
            "The next priority should not be a feature-heavy v7. Freeze the current thresholds, evaluate unseen application-level workloads, repeat on another GPU architecture, "
            "and add measurements from UPMEM or another physical PIM platform before making hardware-level performance claims.",
            s["body"],
        ),
        PageBreak(),
    ]


def _reproduction(s: dict[str, ParagraphStyle]) -> list[object]:
    server_commands = """cd ~/GPU-Workload-Bottleneck-Analyzer
git pull
source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
bash scripts/build_benchmarks.sh
bash scripts/profile_nvprof.sh
NCU_EXTRA_ARGS="--section MemoryWorkloadAnalysis --section SchedulerStats --section WarpStateStats" \\
  bash scripts/profile_ncu.sh
python3 scripts/parse_ncu_reports.py --input-dir profiles/ncu --output profiles/ncu_metrics.csv
python3 scripts/parse_sait_pim_logs.py --log-dir ~/pim-tools/pim-results \\
  --output simulators/sait_pim_simulation.csv --cycle-time-ns 1.0
python3 main.py --input profiles/gpu_profile.csv \\
  --paper-baseline paper_baselines/gpu_benchmark_metadata.csv \\
  --pim-simulation simulators/sait_pim_simulation.csv \\
  --ncu-metrics profiles/ncu_metrics.csv \\
  --output-dir outputs/gpu_profile_with_sait_pim_ncu \\
  --hardware-name "RTX 2080 Ti" --peak-flops 13450000000000 \\
  --peak-memory-bandwidth 616000000000
bash scripts/profile_size_sweep.sh"""
    local_commands = """cd /Users/kingjung/Desktop/gpu-bottleneck-analyzer
git pull
bash scripts/fetch_server_results.sh gpu_profile_with_sait_pim_ncu
open outputs/gpu_profile_with_sait_pim_ncu/reports/analysis_report.md
open outputs/size_sweep/size_sweep_summary.md
open outputs/size_sweep/size_sweep.png
open output/pdf/gpu_pim_bottleneck_analyzer_portfolio_report_en.pdf"""
    return [
        Paragraph("11. Reproduction", s["h1"]),
        Paragraph("11.1 Run on the GPU server", s["h2"]),
        Paragraph(_escape_code(server_commands), s["code"]),
        Paragraph("11.2 Fetch results to the local Mac", s["h2"]),
        Paragraph(_escape_code(local_commands), s["code"]),
        Paragraph("11.3 Success criteria", s["h2"]),
        Paragraph(
            "A complete run contains nine rows in gpu_profile.csv, nine kernels in ncu_metrics.csv, four simulator mappings, three final figures, a Markdown report, "
            "three size-sweep profiles, and the combined size-sweep summary and figure. The test command must report 31 passed.",
            s["callout"],
        ),
        PageBreak(),
    ]


def _conclusion(s: dict[str, ParagraphStyle]) -> list[object]:
    return [
        Paragraph("12. Conclusion", s["h1"]),
        Paragraph(
            "The GPU Workload Bottleneck Analyzer connects GPU performance diagnosis, memory-system counter interpretation, and PIM/NMP candidate selection in one reproducible tool. "
            "Its central contribution is the progression from low-AI rules to a reuse-aware model, an NCU cache/stall-aware v6 policy, and a clock-domain-safe simulator integration.",
            s["body"],
        ),
        Paragraph(
            "Project quality is not defined by the largest speedup number. It is defined by whether the result can be reproduced, traced to evidence, and explained with its limitations. "
            "The repository is complete as a portfolio-grade research prototype. Its next stage is broader external validation, not additional unvalidated model complexity.",
            s["callout"],
        ),
        Paragraph("References", s["h2"]),
        Paragraph(
            "[1] S. Williams, A. Waterman, and D. Patterson, 'Roofline: An Insightful Visual Performance Model for Multicore Architectures,' CACM, 2009.<br/>"
            "[2] NVIDIA, Nsight Compute Kernel Profiling Guide: Roofline, Memory Workload Analysis, and Scheduler Statistics.<br/>"
            "[3] J. Gomez-Luna et al., 'Benchmarking a New Paradigm: An Experimental Analysis of a Real Processing-in-Memory Architecture,' IEEE Access, 2022.<br/>"
            "[4] Samsung Advanced Institute of Technology, SAITPublic/PIMSimulator.<br/>"
            "[5] NVIDIA, CUDA C++ Programming Guide and cuBLAS documentation.",
            s["small"],
        ),
        Spacer(1, 12 * mm),
        _rule(TEAL, 172 * mm),
        Spacer(1, 5 * mm),
        Paragraph(
            "Repository artifact: GPU-Workload-Bottleneck-Analyzer<br/>"
            "Final implementation includes source, tests, raw and parsed profiles, simulator adapters, generated reports, and bilingual portfolio documents.",
            s["small"],
        ),
    ]


def _pipeline_drawing_en() -> Drawing:
    drawing = Drawing(172 * mm, 55 * mm)
    labels = [
        ("GPU Profile", "runtime / bytes"),
        ("Roofline", "AI / utilization"),
        ("NCU + v6", "cache / stalls"),
        ("PIM Sim", "cycles / ratio"),
        ("Decision", "evidence tier"),
    ]
    box_w = 29 * mm
    gap = 6 * mm
    y = 18 * mm
    for index, (title, subtitle) in enumerate(labels):
        x = index * (box_w + gap)
        fill = TEAL if index in {0, 4} else PALE
        text_color = WHITE if index in {0, 4} else INK
        drawing.add(Rect(x, y, box_w, 22 * mm, rx=2 * mm, ry=2 * mm, fillColor=fill, strokeColor=TEAL))
        drawing.add(String(x + box_w / 2, y + 13.5 * mm, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=8, fillColor=text_color))
        drawing.add(String(x + box_w / 2, y + 7 * mm, subtitle, textAnchor="middle", fontName="Helvetica", fontSize=6.2, fillColor=text_color))
        if index < len(labels) - 1:
            x1 = x + box_w
            x2 = x + box_w + gap
            drawing.add(Line(x1 + 1 * mm, y + 11 * mm, x2 - 1 * mm, y + 11 * mm, strokeColor=GOLD, strokeWidth=1.4))
            drawing.add(Line(x2 - 3 * mm, y + 13 * mm, x2 - 1 * mm, y + 11 * mm, strokeColor=GOLD, strokeWidth=1.4))
            drawing.add(Line(x2 - 3 * mm, y + 9 * mm, x2 - 1 * mm, y + 11 * mm, strokeColor=GOLD, strokeWidth=1.4))
    drawing.add(String(86 * mm, 6 * mm, "Measured, simulated, and estimated values remain traceable", textAnchor="middle", fontName="Helvetica", fontSize=8, fillColor=MUTED))
    return drawing


if __name__ == "__main__":
    main()
