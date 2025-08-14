# hints.py
SEED_HINTS = {
  "Congressional Budget Office": {
    "geography": "National scope; CBO does not publish state-by-state OBBB breakouts by default.",
    "disenrollment_projection": (
      "CBO long-run view: ~10.3–10.5M fewer people on Medicaid by 2034 in versions assessed; "
      "about 7.8M of these become fully uninsured due to Medicaid provisions; "
      "total uninsured increase across all health provisions ≈10.9M in the House-passed variant."
    ),
    "funding_impact": (
      "Ten-year federal Medicaid reductions on the order of ~$0.8–$0.9T, with cuts backloaded into 2030–2034; "
      "analysts round total federal health reductions ≈$1T depending on assumptions about caps and phase-ins."
    ),
    "how_enrollment_impact_is_derived": (
      "CBO scores legislative changes (e.g., work requirements, eligibility/verification rules, FMAP/financing, "
      "ACA subsidy expirations) using microsimulation of take-up, churn, and crowd-out effects."
    ),
    "methodology": (
      "CBO microsimulation + program baselines; applies behavioral elasticities, administrative frictions, "
      "and phased policy timing to long-run enrollment and outlay paths."
    ),
    "sources_of_funding_or_backing": "Nonpartisan office of Congress; funded via federal appropriations.",
    "key_differences": (
      "Most comprehensive federal score; results are national and program-wide. Advocacy/think-tank estimates often "
      "reinterpret CBO components (e.g., separating ‘dual-eligibles,’ cross-state counts) to emphasize impact channels."
    ),
    "links_sources": [
      "https://www.cbo.gov",  # CBO home
      "https://rules.house.gov/bill/119/hr-ORH-one-big-beautiful-bill-act",  # rule info
      "https://www.kff.org/medicaid/issue-brief/allocating-cbos-estimates-of-federal-medicaid-spending-reductions-across-the-states-enacted-reconciliation-package/"
    ],
    # citations
    "_refs": ["turn0search15","turn0search13","turn0search9","turn4search14"]
  },

  "Center on Budget & Policy Priorities": {
    "geography": "National analyses with state implications highlighted in commentary.",
    "disenrollment_projection": (
      "Frames coverage losses broadly consistent with CBO totals, emphasizing elevated risk among low-income adults, "
      "disabled people, and seniors; highlights millions potentially losing Medicaid and ACA coverage."
    ),
    "funding_impact": (
      "Emphasizes deep federal Medicaid/ACA cuts and downstream state budget stress; underscores backloaded federal savings."
    ),
    "how_enrollment_impact_is_derived": (
      "Synthesizes CBO and agency data; applies CBPP distributional lenses (income, disability, demographics)."
    ),
    "methodology": (
      "Policy analysis using CBO scores, agency rule text, and prior waiver/work-requirement evidence."
    ),
    "sources_of_funding_or_backing": "Nonprofit think tank; philanthropy and grants.",
    "key_differences": (
      "More distributional/impact-on-vulnerable framing vs. strictly budgetary; aligns closely with CBO but stresses "
      "administrative churn and barriers that can magnify disenrollment."
    ),
    "links_sources": [
      "https://www.cbpp.org/",
      "https://www.cbpp.org/blog/federal-health-bill-would-increase-uninsured-and-cut-medicaid"  # example
    ],
    "_refs": ["turn0search11","turn0search16"]
  },

  "Urban Institute": {
    "geography": "Primarily national with tools to allocate impacts by state when data permit.",
    "disenrollment_projection": (
      "Cites large Medicaid coverage reductions consistent with CBO scoring; provides context with ASPE/Urban prior work "
      "on take-up and churn under tighter eligibility and work verification."
    ),
    "funding_impact": (
      "Notes federal Medicaid outlay reductions and cost shifts to states; emphasizes backloaded nature of cuts."
    ),
    "how_enrollment_impact_is_derived": (
      "Combines CBO baselines with Urban/ASPE evidence on eligibility frictions and work-requirement churn."
    ),
    "methodology": (
      "Microsimulation and literature synthesis; allocation methods sometimes used to state-level breakouts."
    ),
    "sources_of_funding_or_backing": "Independent research institute; foundation and grant support.",
    "key_differences": (
      "Generally consistent with CBO totals; adds implementation mechanics (admin friction, churn) and state allocation tools."
    ),
    "links_sources": [
      "https://www.urban.org/health-policy-center",
      "https://aspe.hhs.gov"  # context for ASPE/Urban evidence
    ],
    "_refs": ["turn1search0","turn1search1"]
  },

  "Kaiser Family Foundation": {
    "geography": "National with state allocation and dashboards.",
    "disenrollment_projection": (
      "Uses CBO totals as baseline; provides state allocation of ten-year Medicaid reductions and explains that ~76% of "
      "reductions occur in 2030–2034 (backloaded effects)."
    ),
    "funding_impact": (
      "Translates CBO’s federal Medicaid outlay reductions into state impacts and provider exposure; highlights the timing."
    ),
    "how_enrollment_impact_is_derived": (
      "Summarizes CBO enrollment and outlay channels; applies KFF allocation methodology to distribute reductions across states."
    ),
    "methodology": (
      "Secondary analysis/visualization of CBO estimates; state allocation model with transparent assumptions."
    ),
    "sources_of_funding_or_backing": "Independent nonprofit; philanthropy and grants.",
    "key_differences": (
      "Most useful for state-level ‘who bears the cut’ view; reconciles CBO totals with state allocation rather than "
      "producing an independent national total."
    ),
    "links_sources": [
      "https://www.kff.org/medicaid/issue-brief/allocating-cbos-estimates-of-federal-medicaid-spending-reductions-across-the-states-enacted-reconciliation-package/"
    ],
    "_refs": ["turn4search14","turn0search9","turn0search5","turn0search7"]
  },

  "CA Governor’s Office": {
    "geography": "State of California.",
    "disenrollment_projection": (
      "California leadership warns of large Medi-Cal and Covered California coverage losses; California Medical Association "
      "cites ~2.5M Medi-Cal and up to 2.6M exchange enrollees at risk under OBBB-style changes."
    ),
    "funding_impact": (
      "State budget stress from reduced federal match and ACA subsidies; administration signaled offsets/guardrails where possible "
      "but acknowledged fiscal constraints."
    ),
    "how_enrollment_impact_is_derived": (
      "Draws on state budget modeling and external estimates (CMA, KFF, advocacy groups) pending official DHCS breakouts."
    ),
    "methodology": (
      "Executive/budget office analysis; relies on CBO baselines + state fiscal/budget offices; details to be refined by DHCS."
    ),
    "sources_of_funding_or_backing": "State general fund, special funds (e.g., provider tax), federal match; constrained under OBBB.",
    "key_differences": (
      "California-specific lens emphasizing Medi-Cal/ACA impacts, state backfills, and legal/policy countermeasures; "
      "focus on program operations and budget maneuvers rather than national totals."
    ),
    "links_sources": [
      "https://www.cmadocs.org/newsroom/news/view/ArticleId/50933/-Big-Beautiful-Bill-will-devastate-access-to-care-CMA-warns",
      "https://calbudgetcenter.org/resources/state-leaders-must-boldly-respond-to-the-devastating-cuts-in-president-trumps-budget-bill/",
      "https://www.courthousenews.com/gavin-newsom-sends-up-warning-flag-over-big-beautiful-trump-budget/"
    ],
    "_refs": ["turn4search8","turn4search7","turn4search11"]
  }
}
