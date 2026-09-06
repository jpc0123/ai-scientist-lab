import { Component, type ErrorInfo, type ReactNode } from "react";

type Props = { children: ReactNode };
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Stack stays in console / logs — not rendered as a raw traceback panel.
    console.error("UI error boundary", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="api-error-view error-panel" role="alert">
          <h2>页面出错</h2>
          <dl className="api-error-dl">
            <div>
              <dt>错误是什么</dt>
              <dd>{this.state.error.message || "未知渲染错误"}</dd>
            </div>
            <div>
              <dt>为什么发生</dt>
              <dd>前端组件渲染或状态更新失败（堆栈仅在浏览器控制台）。</dd>
            </div>
            <div>
              <dt>你可以做什么</dt>
              <dd>点击重试；若反复出现，刷新页面或回到总览。</dd>
            </div>
            <div>
              <dt>是否可重试</dt>
              <dd>可以重试</dd>
            </div>
          </dl>
          <button type="button" onClick={() => this.setState({ error: null })}>
            重试
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
