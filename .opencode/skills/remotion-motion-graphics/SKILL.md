---
name: Remotion Motion Graphics
description: Create and edit Remotion compositions, animated videos, previews, and renders in the isolated motion-graphics project.
---

# Remotion motion graphics

- Keep Remotion code and Node dependencies inside `motion-graphics/`. Do not add
  JavaScript dependencies to the FinOlimp reconciliation project or modify its
  accounting inputs and reports for a graphics task.
- Before creating or editing a composition, read
  `motion-graphics/.agents/skills/remotion-best-practices/SKILL.md`. Follow its
  linked task-specific references, especially `remotion-create` for a new video,
  `remotion-markup` for React layout, and `remotion-render` for export details.
- The starter app is TypeScript with Remotion 4.0.534. Preserve the existing
  `src/index.ts`, `src/Root.tsx`, and composition registration pattern unless
  the requested change requires otherwise.
- Preview with `cd motion-graphics && npm run dev`. Run
  `cd motion-graphics && npm run lint` after code changes.
- Do not render or export a final video unless the user explicitly asks. If
  rendering is requested, follow the official render skill and save outputs
  inside `motion-graphics/out/`.
