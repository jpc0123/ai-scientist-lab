import { useParams } from "react-router-dom";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "../components/AppShell";
import { ApprovalsPage } from "../pages/ApprovalsPage";
import { DashboardPage } from "../pages/DashboardPage";
import { ExecutionDetailPage, ExecutionsPage } from "../pages/ExecutionsPage";
import { ComparePage } from "../pages/ComparePage";
import { PatchDetailPage, PatchesPage } from "../pages/PatchesPage";
import { ProjectCreateWizardPage, ProjectDetailPage, ProjectsPage } from "../pages/ProjectsPage";
import {
  AuditDetailPage,
  AuditsPage,
  ReportDetailPage,
  ReportsPage,
} from "../pages/ReportsAuditsPages";
import {
  ClaimsPage,
  EvidenceDetailPage,
  EvidencePage,
} from "../pages/EvidenceClaimsPages";
import { PlanDetailPage, PlansPage } from "../pages/PlanningPages";
import {
  IterationDetailPage,
  IterationsPage,
} from "../pages/ResourcePages";
import { SettingsPage } from "../pages/SystemPage";
import { GuidePage } from "../pages/GuidePage";
import { ChatWorkspacePage } from "../pages/ChatWorkspacePage";
import {
  MergeDetailPage,
  MergesPage,
  ReleaseCandidateDetailPage,
  ReleaseCandidatesPage,
  RollbacksPage,
} from "../pages/MergeReleasePages";
import { TreeDetailPage, TreesPage } from "../pages/TreesPage";
import { LlmConfigPage } from "../pages/LlmConfigPage";
import { RealLoopDetailPage, RealLoopsPage } from "../pages/RealLoopsPage";

export function AppRouter() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/assistant" replace />} />
        <Route path="assistant" element={<ChatWorkspacePage />} />
        <Route path="guide" element={<GuidePage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="approvals" element={<ApprovalsPage />} />
        <Route path="projects" element={<ProjectsPage />} />
        <Route path="projects/new" element={<ProjectCreateWizardPage />} />
        <Route path="projects/:projectId" element={<ProjectDetailRoute />} />
        <Route path="projects/:projectId/real-loops" element={<ProjectRealLoopsRoute />} />
        <Route path="executions" element={<ExecutionsPage />} />
        <Route path="executions/:id" element={<ExecutionDetailPage />} />
        <Route path="compare" element={<ComparePage />} />
        <Route path="trees" element={<TreesPage />} />
        <Route path="trees/:id" element={<TreeDetailPage />} />
        <Route path="real-loops" element={<RealLoopsPage />} />
        <Route path="real-loops/:sessionId" element={<RealLoopDetailPage />} />
        <Route path="plans" element={<PlansPage />} />
        <Route path="plans/:id" element={<PlanDetailPage />} />
        <Route path="iterations" element={<IterationsPage />} />
        <Route path="iterations/:id" element={<IterationDetailPage />} />
        <Route path="patches" element={<PatchesPage />} />
        <Route path="patches/:id" element={<PatchDetailPage />} />
        <Route path="merges" element={<MergesPage />} />
        <Route path="merges/:id" element={<MergeDetailPage />} />
        <Route path="rollbacks" element={<RollbacksPage />} />
        <Route path="release-candidates" element={<ReleaseCandidatesPage />} />
        <Route
          path="release-candidates/:id"
          element={<ReleaseCandidateDetailPage />}
        />
        <Route path="evidence" element={<EvidencePage />} />
        <Route path="evidence/:id" element={<EvidenceDetailPage />} />
        <Route path="claims" element={<ClaimsPage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="reports/:id" element={<ReportDetailPage />} />
        <Route path="audits" element={<AuditsPage />} />
        <Route path="audits/:id" element={<AuditDetailPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="llm-config" element={<LlmConfigPage />} />
        <Route path="*" element={<Navigate to="/assistant" replace />} />
      </Route>
    </Routes>
  );
}

function ProjectDetailRoute() {
  const { projectId = "" } = useParams();
  return <ProjectDetailPage projectId={projectId} />;
}

function ProjectRealLoopsRoute() {
  const { projectId = "" } = useParams();
  return <Navigate to={`/real-loops?project_id=${encodeURIComponent(projectId)}`} replace />;
}
