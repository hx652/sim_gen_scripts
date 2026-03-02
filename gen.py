from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set
import xml.sax.saxutils as xml_escape


def _escape_attr(value: Any) -> str:
    """Convert a Python value to a safe XML attribute string."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, (list, tuple)):
        text = " ".join(str(v) for v in value)
    else:
        text = str(value)
    return xml_escape.escape(text, {'"': "&quot;"})


def _format_vec3(values: Sequence[float]) -> str:
    if len(values) != 3:
        raise ValueError(f"Expected length 3, got {values}")
    return " ".join(str(v) for v in values)


@dataclass(frozen=True)
class MacroSpec:
    """Describe which xacro macro should be used for an Arm instance."""
    include_file: str
    macro_name: str


class Link:
    """
    A simple scene link.

    First version:
    - it has a name
    - it is mounted to a parent Link
    - it has an origin relative to its parent
    - it will be rendered as:
        <link name="..."/>
        <joint type="fixed"> ... <origin .../> ... </joint>
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Link] = None,
        xyz: Sequence[float] = (0.0, 0.0, 0.0),
        rpy: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> None:
        if not name:
            raise ValueError("Link.name cannot be empty")
        if len(xyz) != 3:
            raise ValueError(f"Link xyz must have length 3, got {xyz}")
        if len(rpy) != 3:
            raise ValueError(f"Link rpy must have length 3, got {rpy}")

        self.name: str = name
        self.parent: Optional[Link] = parent
        self.xyz: Sequence[float] = xyz
        self.rpy: Sequence[float] = rpy

    def is_root(self) -> bool:
        return self.parent is None

    def origin_attrs(self) -> Dict[str, str]:
        return {
            "xyz": _format_vec3(self.xyz),
            "rpy": _format_vec3(self.rpy),
        }


WORLD = Link("world", parent=None)


class Arm:
    """
    A concrete arm instance in the scene.

    This class stores the information required to instantiate
    an existing xacro macro.
    """

    def __init__(
        self,
        name: str,
        macro: MacroSpec,
        parent: Link,
        xyz: Sequence[float] = (0.0, 0.0, 0.0),
        rpy: Sequence[float] = (0.0, 0.0, 0.0),
        prefix: Optional[str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not name:
            raise ValueError("Arm.name cannot be empty")
        if len(xyz) != 3:
            raise ValueError(f"xyz must have length 3, got {xyz}")
        if len(rpy) != 3:
            raise ValueError(f"rpy must have length 3, got {rpy}")

        self.name: str = name
        self.macro: MacroSpec = macro
        self.parent: Link = parent
        self.xyz: Sequence[float] = xyz
        self.rpy: Sequence[float] = rpy
        self.prefix: str = prefix if prefix is not None else f"{name}_"
        self.extra_args: Dict[str, Any] = extra_args if extra_args is not None else {}

        if not self.prefix:
            raise ValueError("Arm.prefix cannot be empty")

    def to_ordinary_args(self) -> Dict[str, Any]:
        """
        Collect ordinary xacro arguments.

        Example:
            name="arm_0" prefix="arm_0_" parent="world"
        """
        args: Dict[str, Any] = {
            "name": self.name,
            "prefix": self.prefix,
            "parent": self.parent.name,
        }
        args.update(self.extra_args)
        return args

    def to_block_args(self) -> Dict[str, Dict[str, str]]:
        """
        Collect block arguments.

        This version uses one origin block:
            <origin xyz="..." rpy="..."/>
        """
        return {
            "origin": {
                "xyz": _format_vec3(self.xyz),
                "rpy": _format_vec3(self.rpy),
            }
        }


class Scene:
    """A scene containing links and arms."""

    def __init__(self, name: str, root_link: Link = WORLD) -> None:
        if not name:
            raise ValueError("Scene.name cannot be empty")

        self.name: str = name
        self.root_link: Link = root_link
        self.links: List[Link] = []
        self.arms: List[Arm] = []

    def add_link(self, link: Link) -> None:
        if link.name == self.root_link.name:
            raise ValueError(f"Link name conflicts with root link: {link.name}")
        if self.find_link(link.name) is not None:
            raise ValueError(f"Duplicate link name: {link.name}")
        self.links.append(link)

    def add_arm(self, arm: Arm) -> None:
        if self.find_arm(arm.name) is not None:
            raise ValueError(f"Duplicate arm name: {arm.name}")
        self.arms.append(arm)

    def find_link(self, name: str) -> Optional[Link]:
        if self.root_link.name == name:
            return self.root_link
        for link in self.links:
            if link.name == name:
                return link
        return None

    def find_arm(self, name: str) -> Optional[Arm]:
        for arm in self.arms:
            if arm.name == name:
                return arm
        return None

    def validate(self) -> None:
        arm_names: Set[str] = set()
        prefixes: Set[str] = set()
        link_names: Set[str] = {self.root_link.name}

        for link in self.links:
            if link.name in link_names:
                raise ValueError(f"Duplicate link name: {link.name}")
            link_names.add(link.name)

        for link in self.links:
            if link.parent is None:
                raise ValueError(
                    f"Scene link '{link.name}' must have a parent; only root_link may be parentless"
                )
            if link.parent.name not in link_names:
                raise ValueError(
                    f"Link '{link.name}' references unknown parent link '{link.parent.name}'"
                )

        for arm in self.arms:
            if arm.name in arm_names:
                raise ValueError(f"Duplicate arm name: {arm.name}")
            arm_names.add(arm.name)

            if arm.prefix in prefixes:
                raise ValueError(f"Duplicate arm prefix: {arm.prefix}")
            prefixes.add(arm.prefix)

            if arm.parent.name not in link_names:
                raise ValueError(
                    f"Arm '{arm.name}' references unknown parent link '{arm.parent.name}'"
                )


class XacroRenderer:
    """
    Render a Scene into a xacro file.

    This version:
    - collects unique include files from all arms
    - renders scene links
    - renders one xacro macro call per arm
    - handles ordinary arguments and block arguments separately
    """

    XML_HEADER = '<?xml version="1.0"?>'

    def render(self, scene: Scene) -> str:
        scene.validate()

        includes: List[str] = self._collect_includes(scene)
        lines: List[str] = []

        lines.append(self.XML_HEADER)
        lines.append(
            '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" '
            f'name="{_escape_attr(scene.name)}">'
        )
        lines.append("")

        for include_file in includes:
            lines.append(f'  <xacro:include filename="{_escape_attr(include_file)}" />')

        if includes:
            lines.append("")

        lines.extend(self._render_root_link(scene.root_link))
        lines.append("")

        for link in scene.links:
            lines.extend(self._render_link(link))
            lines.append("")

        for arm in scene.arms:
            lines.extend(self._render_arm(arm))
            lines.append("")

        lines.append("</robot>")
        return "\n".join(lines)

    def write(self, scene: Scene, output_path: str | Path) -> None:
        content: str = self.render(scene)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")

    def _collect_includes(self, scene: Scene) -> List[str]:
        seen: Set[str] = set()
        includes: List[str] = []

        for arm in scene.arms:
            include_file = arm.macro.include_file
            if include_file not in seen:
                seen.add(include_file)
                includes.append(include_file)

        return includes

    def _render_root_link(self, root_link: Link) -> List[str]:
        return [f'  <link name="{_escape_attr(root_link.name)}" />']

    def _render_link(self, link: Link) -> List[str]:
        if link.parent is None:
            raise ValueError(f"Non-root link '{link.name}' must have a parent")

        lines: List[str] = []
        lines.append(f'  <link name="{_escape_attr(link.name)}" />')
        lines.append(
            f'  <joint name="{_escape_attr(link.name)}_joint" type="fixed">'
        )
        lines.append(f'    <parent link="{_escape_attr(link.parent.name)}" />')
        lines.append(f'    <child link="{_escape_attr(link.name)}" />')

        origin = link.origin_attrs()
        lines.append(
            f'    <origin xyz="{_escape_attr(origin["xyz"])}" '
            f'rpy="{_escape_attr(origin["rpy"])}" />'
        )
        lines.append("  </joint>")
        return lines

    def _render_arm(self, arm: Arm) -> List[str]:
        ordinary_args: Dict[str, Any] = arm.to_ordinary_args()
        block_args: Dict[str, Dict[str, str]] = arm.to_block_args()

        ordinary_attrs: str = " ".join(
            f'{key}="{_escape_attr(value)}"'
            for key, value in ordinary_args.items()
        )

        if not block_args:
            return [f"  <xacro:{arm.macro.macro_name} {ordinary_attrs} />"]

        lines: List[str] = [f"  <xacro:{arm.macro.macro_name} {ordinary_attrs}>"]

        for block_name, block_attrs in block_args.items():
            attrs: str = " ".join(
                f'{key}="{_escape_attr(value)}"'
                for key, value in block_attrs.items()
            )
            lines.append(f"    <{block_name} {attrs} />")

        lines.append(f"  </xacro:{arm.macro.macro_name}>")
        return lines


if __name__ == "__main__":
    arm_macro = MacroSpec(
        include_file="$(find my_robot_description)/urdf/arm_macro.xacro",
        macro_name="my_arm",
    )

    scene = Scene(name="multi_arm_scene")

    scene.add_arm(
        Arm(
            name="arm_0",
            macro=arm_macro,
            parent=WORLD,
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 0.0),
        )
    )

    scene.add_arm(
        Arm(
            name="arm_1",
            macro=arm_macro,
            parent=WORLD,
            xyz=(0.0, 0.0, 0.0),
            rpy=(0.0, 0.0, 1.57),
            extra_args={
                "package_name": "kuka_quantec_support"
            },
        )
    )

    renderer = XacroRenderer()
    renderer.write(scene, "generated/multi_arm_scene.xacro")
    print("Generated: generated/multi_arm_scene.xacro")