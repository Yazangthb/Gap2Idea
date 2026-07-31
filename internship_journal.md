# Internship Journal — Gap2Idea

**Period:** 01/12/2026 → 05/23/2026
**Project:** Gap2Idea — LLM-based pipeline for extracting research gaps from scientific papers and generating new research ideas.

---

## Individual task

**RU:** Разработать сквозную систему Gap2Idea для извлечения исследовательских пробелов из научных статей и автоматической генерации новых исследовательских идей с помощью больших языковых моделей.

**EN:** Design and build the end-to-end Gap2Idea pipeline that extracts research gaps from scientific papers and automatically generates new research ideas using LLMs.

---

## Expected results

**RU:** Рабочий конвейер: загрузка статей → извлечение пробелов и методов → тематическая кластеризация → генерация идей (несколько режимов) → LLM-оценка → экспорт LaTeX/PDF; бенчмарки качества; MCP-сервер; облачное развёртывание.

**EN:** A working pipeline: paper ingestion → gap/method extraction → theme clustering → multi-mode idea generation → LLM evaluation → LaTeX/PDF export; quality benchmarks; MCP server; cloud deployment.

---

## Brief description of achieved result

**RU:** Реализован Gap2Idea: извлечение пробелов/методов из PDF, эмбеддинги и кластеризация, 4 режима генерации идей (bridge, within, method-gap, multi-agent), оценка LLM-жюри, экспорт в LaTeX/PDF, генератор полных черновиков статей, MCP-сервер для Claude Desktop, бенчмарки извлечения и кластеризации, развёртывание на Google Cloud Run.

**EN:** Delivered Gap2Idea: extracts gaps and methods from PDFs, clusters by theme, generates ideas in 4 modes (bridge, within, method-gap, multi-agent), evaluates with an LLM judge panel, exports per-idea LaTeX/PDF, expands ideas into full paper drafts, exposes artifacts via an MCP server, ships extraction and clustering benchmarks, and deploys to Google Cloud Run.

---

## Self-reflection

**RU:** Освоил оркестрацию LLM, проектирование бенчмарков (vs. unarXive gold) и инженерию сквозных ML-систем. Основная сложность — объективная оценка новизны идей; решено через S2 novelty + аудит пересечения с источниками + панель судей. Понял ценность ранней автоматизации и воспроизводимости (uv, Docker, ablation-плоты).

**EN:** Learned LLM orchestration, benchmark design (against unarXive gold sections), and end-to-end ML system engineering. Hardest part was objectively scoring idea novelty — solved via S2 novelty check, evidence-overlap audit, and an LLM judge panel. Saw the value of early automation and reproducibility (uv, Docker, ablation plots).

---

# Journal records

## Stage 1 — 01/12/2026 → 01/27/2026

**Internship task RU:** Настроить инфраструктуру проекта, Docker, uv и конвейер загрузки научных статей.

**Internship task EN:** Set up project infrastructure, Docker, uv, and the paper ingestion pipeline.

**Internship results RU:** Рабочая платформа, Docker/uv окружение, конвейер загрузки PDF, начальная документация.

**Internship results EN:** Working platform, Docker/uv environment, PDF ingestion pipeline, initial documentation.

---

## Stage 2 — 01/28/2026 → 02/28/2026

**Internship task RU:** Реализовать семантический поиск по статьям и извлечение структурных секций из PDF.

**Internship task EN:** Implement semantic search over papers and structural section extraction from PDFs.

**Internship results RU:** Семантический поиск по корпусу статей, парсер секций PDF, эмбеддинги документов.

**Internship results EN:** Semantic search over the paper corpus, PDF section parser, document embeddings.

---

## Stage 3 — 03/01/2026 → 05/15/2026

**Internship task RU:** Создать полный конвейер генерации и оценки исследовательских идей на базе LLM.

**Internship task EN:** Build the full pipeline for LLM-based research idea generation and evaluation.

**Internship results RU:** Извлечение методов, 4 режима генерации идей, LLM-жюри, MCP-сервер, мультиагентный поток.

**Internship results EN:** Methods extraction, 4 idea-generation modes, LLM judge panel, MCP server, multi-agent flow.

---

## Stage 4 — 05/16/2026 → 05/23/2026

**Internship task RU:** Провести бенчмарки качества, реализовать генератор статей и облачное развёртывание.

**Internship task EN:** Run quality benchmarks, implement the paper drafter, and set up cloud deployment.

**Internship results RU:** Бенчмарки извлечения и кластеризации, генератор черновиков статей, Cloud Run-деплой.

**Internship results EN:** Extraction and clustering benchmarks, full paper-draft generator, Cloud Run deployment.

---

**Knowledge gained:** Significantly improved
**Internship organization quality:** Excellent

---

# Supervisor feedback (university)

## Quality of work and result satisfaction

**RU:** Работа выполнена на высоком уровне. Студент самостоятельно спроектировал и реализовал сквозной конвейер Gap2Idea: извлечение пробелов и методов из научных PDF, эмбеддинги и тематическая кластеризация, четыре режима генерации идей (включая мультиагентный), оценка LLM-жюри, экспорт в LaTeX/PDF, генератор полных черновиков статей, MCP-сервер и развёртывание на Google Cloud Run. Все заявленные цели достигнуты и подкреплены бенчмарками качества извлечения (vs. unarXive gold) и кластеризации с ablation-исследованиями. Код хорошо структурирован, воспроизводим (uv, Docker), сопровождается документацией и диаграммами. Результатами полностью удовлетворён.

**EN:** The work was carried out at a high level. The student independently designed and implemented the end-to-end Gap2Idea pipeline: gap and method extraction from scientific PDFs, embeddings and theme clustering, four idea-generation modes (including multi-agent), LLM judge-panel evaluation, LaTeX/PDF export, a full paper-draft generator, an MCP server, and Google Cloud Run deployment. All stated goals were met and backed by extraction-quality benchmarks (against unarXive gold sections) and clustering benchmarks with ablation studies. The code is well-structured, reproducible (uv, Docker), and accompanied by documentation and diagrams. Fully satisfied with the results.

---

## General recommendations

**RU:** Рекомендуется продолжить развитие проекта в сторону масштабирования: расширить корпус за пределы arXiv-снимка, добавить более строгую человеческую валидацию новизны идей (помимо LLM-жюри и S2-проверки), и провести пользовательское исследование с исследователями-доменными экспертами. Стоит также подготовить публикацию по методологии оценки идей и оформить мультиагентный модуль как отдельный воспроизводимый бенчмарк. В дальнейшем — уделить внимание мониторингу стоимости LLM-вызовов и кешированию промежуточных артефактов в продакшене.

**EN:** It is recommended to continue developing the project toward scale: extend the corpus beyond the arXiv snapshot, add stricter human validation of idea novelty (alongside the LLM judge panel and S2 check), and run a user study with domain-expert researchers. A publication on the idea-evaluation methodology would be worthwhile, as would packaging the multi-agent module as a standalone reproducible benchmark. Going forward, attention should be paid to monitoring LLM-call costs and caching intermediate artifacts in production.
