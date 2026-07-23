import { useParams } from "react-router-dom";
import { Navigate, Route, Routes } from "react-router-dom";
import { AppShell } from "../components/AppShell";
import { ApprovalsPage } from "../pages/ApprovalsPage";
import { DashboardPage } from "../pages/DashboardPage";
import { ExecutionDetailPage, ExecutionsPage } from "../pages/ExecutionsPage";
import { PatchDetailPage, PatchesPage } from "../pages/PatchesPage";
import { ProjectDetailPage, ProjectsPage } from "../pages/ProjectsPage";
import {
  AuditDetailPage,
  AuditsPage,
  EvidencePage,
  ReportDetailPage,
  ReportsPage,
} from "../pages/ReportsAuditsPages";
import {
  IterationDetailPage,
  IterationsPage,
  PlanDetailPage,
  PlansPage,
  SettingsPage,
} from "../pages/ResourcePages";
import { GuidePage } from "../pages/GuidePage";
import { TreeDetailPage, TreesPage } from "../pages/TreesPage";

export function AppRouter() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="guide" element={<GuidePage />} />
        <Route path="dashboard" element={<DashboardPage />} />
        <Route path="approvals" element={<ApprovalsPage />} />
        <Route path="projects" element={<ProjectsPage />} />
        <Route path="projects/:projectId" element={<ProjectDetailRoute />} />
        <Route path="executions" element={<ExecutionsPage />} />
        <Route path="executions/:id" element={<ExecutionDetailPage />} />
        <Route path="trees" element={<TreesPage />} />
        <Route path="trees/:id" element={<TreeDetailPage />} />
        <Route path="plans" element={<PlansPage />} />
        <Route path="plans/:id" element={<PlanDetailPage />} />
        <Route path="iterations" element={<IterationsPage />} />
        <Route path="iterations/:id" element={<IterationDetailPage />} />
        <Route path="patches" element={<PatchesPage />} />
        <Route path="patches/:id" element={<PatchDetailPage />} />
        <Route path="evidence" element={<EvidencePage />} />
        <Route path="reports" element={<ReportsPage />} />
        <Route path="reports/:id" element={<ReportDetailPage />} />
        <Route path="audits" element={<AuditsPage />} />
        <Route path="audits/:id" element={<AuditDetailPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}

function ProjectDetailRoute() {
  const { projectId = "" } = useParams();
  return <ProjectDetailPage projectId={projectId} />;
}
