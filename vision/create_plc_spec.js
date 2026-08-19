import fs from "node:fs";
import path from "node:path";
import {
  AlignmentType,
  Document,
  Footer,
  Header,
  HeadingLevel,
  ImportedXmlComponent,
  Packer,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
  convertInchesToTwip,
} from "docx";

const outputPath = process.argv[2];
if (!outputPath) {
  throw new Error("Usage: node create_plc_spec.js /absolute/path/output.docx");
}

const T = String.raw;

const palette = {
  dark: "263238",
  primary: "37474F",
  light: "78909C",
  border: "D8E0E3",
  fill: "EEF3F6",
  red: "C62828",
  green: "2E7D32",
};

const font = {
  ascii: "Times New Roman",
  hAnsi: "Times New Roman",
  cs: "Times New Roman",
  eastAsia: "SimSun",
};

const run = (text, options = {}) =>
  new TextRun({ text, font, size: 24, ...options });

const para = (children, options = {}) =>
  new Paragraph({
    spacing: { after: 160, line: 300 },
    ...options,
    children: Array.isArray(children) ? children : [children],
  });

const bodyPara = (text, options = {}) =>
  para(run(text), {
    indent: { firstLine: convertInchesToTwip(0.33) },
    ...options,
  });

const heading = (text, level = 1) =>
  para(run(text, { bold: true, size: level === 1 ? 30 : level === 2 ? 26 : 24, color: palette.dark }), {
    heading: level === 1 ? HeadingLevel.HEADING_1 : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
    spacing: { before: level === 1 ? 360 : 240, after: 120 },
  });

const cell = (text, options = {}) =>
  new TableCell({
    children: [para(run(text, { size: 22 }), { spacing: { after: 80, line: 260 } })],
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    ...options,
  });

const xmlEscape = (value) =>
  String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");

const toc = (entries) => {
  const cached = entries
    .map(({ title: entryTitle, level, page }) => {
      const indent = Math.max(0, level - 1) * 360;
      return `<w:p>
        <w:pPr>
          <w:pStyle w:val="TOC${level}"/>
          <w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9000"/></w:tabs>
          <w:ind w:left="${indent}"/>
        </w:pPr>
        <w:r><w:t>${xmlEscape(entryTitle)}</w:t></w:r>
        <w:r><w:tab/></w:r>
        <w:r><w:t>${xmlEscape(page)}</w:t></w:r>
      </w:p>`;
    })
    .join("");

  return ImportedXmlComponent.fromXmlString(`<w:sdt xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:sdtPr><w:alias w:val="目录"/></w:sdtPr>
    <w:sdtContent>
      <w:p>
        <w:r>
          <w:fldChar w:fldCharType="begin" w:dirty="true"/>
          <w:instrText xml:space="preserve"> TOC \\o &quot;1-3&quot; \\h \\z \\u </w:instrText>
          <w:fldChar w:fldCharType="separate"/>
        </w:r>
      </w:p>
      ${cached}
      <w:p><w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>
    </w:sdtContent>
  </w:sdt>`).root[0];
};

// ===== 表格构建器 =====
const makeTable = (headers, rows, colWidths) => {
  const headerCells = headers.map((h, i) =>
    cell(h, {
      shading: { type: ShadingType.CLEAR, fill: palette.fill },
      width: { size: colWidths[i], type: WidthType.DXA },
    })
  );
  const dataRows = rows.map((row) =>
    new TableRow({
      children: row.map((text, i) =>
        cell(text, { width: { size: colWidths[i], type: WidthType.DXA } })
      ),
    })
  );
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: colWidths,
    rows: [
      new TableRow({ children: headerCells }),
      ...dataRows,
    ],
  });
};

// ===== 文档内容 =====
const docTitle = T`视觉系统 PLC 接口规范文档`;
const docSubtitle = T`Modbus TCP 通讯协议 — 缺陷分类视觉检测系统`;
const docVersion = T`版本 v2.3 | 2026-08-19`;

const sections = [
  {
    title: T`一、概述`,
    level: 1,
    page: 2,
    content: [
      {
        type: "text",
        text: T`本文档定义视觉检测系统与 PLC 之间的 Modbus TCP 通讯接口规范，适用于视觉逻辑算法应用赛缺陷分类项目。双方通过 Modbus TCP 协议交换触发信号与检测结果，实现物料缺陷的自动识别与分拣控制。`,
      },
      { type: "h2", text: T`1.1 通讯协议参数` },
      {
        type: "table",
        headers: [T`参数项`, T`设定值`, T`说明`],
        rows: [
          [T`通讯协议`, T`Modbus TCP`, T`赛题要求，主路径`],
          [T`PLC IP 地址`, T`192.168.3.99`, T`现场可调整`],
          [T`端口号`, T`502`, T`Modbus TCP 标准端口`],
          [T`从站号 (unit_id)`, T`1`, T`PLC 端需配置相同`],
          [T`超时时间`, T`2.0 秒`, T`单次读写超时`],
          [T`通讯频率`, T`每工件 1 次`, T`5 分钟内 12 工件`],
        ],
        widths: [2200, 2200, 3200],
      },
      { type: "h2", text: T`1.2 网络拓扑` },
      {
        type: "text",
        text: T`视觉系统以 Docker 容器形式运行于 EPC 工控机，通过 host 网络模式直接访问 PLC 的 Modbus TCP Server。容器与 PLC 处于同一局域网段，无需额外的端口映射或 NAT 配置。`,
      },
    ],
  },
  {
    title: T`二、寄存器映射表`,
    level: 1,
    page: 3,
    content: [
      {
        type: "text",
        text: T`以下保持寄存器（Holding Register）地址为默认配置，可在视觉端 config.yaml 中调整。若 PLC 端已有固定地址规划，双方协商后由视觉侧修改配置即可。`,
      },
      {
        type: "table",
        headers: [T`寄存器地址`, T`类型`, T`方向`, T`功能`, T`数据格式`],
        rows: [
          [T`100`, T`Holding Register`, T`PLC → Vision`, T`触发信号 xVisionTrigger`, T`0 = 无触发 / 1 = 触发拍照`],
          [T`110`, T`Holding Register`, T`Vision → PLC`, T`缺陷类别 xDefectClass`, T`1=hole, 2=chip, 3=scratch, 4=stain, 0=未识别`],
          [T`111`, T`Holding Register`, T`Vision → PLC`, T`置信度 fConfidence`, T`0~1000（实际值 × 1000）`],
          [T`112`, T`Holding Register`, T`Vision → PLC`, T`结果有效 xResultValid`, T`0 = 无效 / 1 = 有效`],
          [T`113`, T`Holding Register`, T`Vision → PLC`, T`序列号 iResultSeq`, T`递增整数，PLC 可校验丢包`],
        ],
        widths: [1300, 1700, 1600, 1900, 2100],
      },
      {
        type: "text",
        text: T`注：视觉端已对缺陷类别做了 +1 偏移处理（0 保留给未识别），PLC 收到的值直接对应 1~4，无需再做偏移计算。`,
      },
    ],
  },
  {
    title: T`三、PLC 端配置步骤`,
    level: 1,
    page: 4,
    content: [
      { type: "h2", text: T`3.1 变量定义` },
      {
        type: "text",
        text: T`在 PLCnext Engineer 的 MAIN 程序中定义以下 5 个 INT 类型变量，并与对应的保持寄存器绑定：`,
      },
      {
        type: "table",
        headers: [T`PLC 变量名`, T`数据类型`, T`对应寄存器`, T`方向`],
        rows: [
          [T`xVisionTrigger`, T`INT`, T`HR 100`, T`PLC → Vision`],
          [T`xDefectClass`, T`INT`, T`HR 110`, T`Vision → PLC`],
          [T`fConfidence`, T`INT`, T`HR 111`, T`Vision → PLC`],
          [T`xResultValid`, T`INT`, T`HR 112`, T`Vision → PLC`],
          [T`iResultSeq`, T`INT`, T`HR 113`, T`Vision → PLC`],
        ],
        widths: [2200, 1600, 2000, 2200],
      },
      { type: "h2", text: T`3.2 启用 Modbus TCP Server` },
      {
        type: "list",
        items: [
          T`在 PLCnext 中启用 Modbus TCP Server 功能`,
          T`设置监听端口：502`,
          T`设置从站号 (Unit ID)：1`,
          T`将上述 5 个变量映射到对应的保持寄存器地址`,
        ],
      },
      { type: "h2", text: T`3.3 触发逻辑` },
      {
        type: "text",
        text: T`PLC 在检测到物料到达工位后，将 HR[100] 置为 1，视觉系统检测到触发信号后执行拍照、推理，完成后将结果写入 HR[110]~HR[113]。PLC 在确认 valid=1 后读取结果并控制三轴执行分类放置，随后将 HR[100] 清零，等待下一次触发。`,
      },
    ],
  },
  {
    title: T`四、数据通讯时序`,
    level: 1,
    page: 5,
    content: [
      {
        type: "text",
        text: T`单次完整的检测周期时序如下：`,
      },
      {
        type: "table",
        headers: [T`步骤`, T`发起方`, T`操作`, T`寄存器`, T`数值示例`],
        rows: [
          [T`1`, T`PLC`, T`物料到达，触发视觉`, T`HR[100]`, T`1`],
          [T`2`, T`Vision`, T`读取触发信号`, T`HR[100]`, T`1`],
          [T`3`, T`Vision`, T`抓帧`, T`-`, T`-`],
          [T`4`, T`Vision`, T`图像预处理`, T`-`, T`-`],
          [T`5`, T`Vision`, T`模型推理`, T`-`, T`-`],
          [T`6`, T`Vision`, T`写入缺陷类别`, T`HR[110]`, T`3 (scratch)`],
          [T`7`, T`Vision`, T`写入置信度`, T`HR[111]`, T`920 (0.92)`],
          [T`8`, T`Vision`, T`写入有效标志`, T`HR[112]`, T`1`],
          [T`9`, T`Vision`, T`写入序列号`, T`HR[113]`, T`N`],
          [T`10`, T`PLC`, T`读取结果，执行搬运`, T`HR[110~113]`, T`-`],
          [T`11`, T`PLC`, T`清零触发`, T`HR[100]`, T`0`],
        ],
        widths: [900, 1400, 2200, 1600, 2100],
      },
      {
        type: "text",
        text: T`正常单次检测周期约为 70ms（视觉端推理 26ms + 通讯开销），远小于赛题 3 分钟的总时限要求。`,
      },
    ],
  },
  {
    title: T`五、缺陷类别映射表`,
    level: 1,
    page: 6,
    content: [
      {
        type: "text",
        text: T`视觉系统识别 4 类缺陷，通过 HR[110] 上报给 PLC。PLC 根据该值控制三轴将物料放置到对应工位。`,
      },
      {
        type: "table",
        headers: [T`HR[110] 值`, T`缺陷名称`, T`英文标识`, T`PLC 建议动作`],
        rows: [
          [T`1`, T`孔洞`, T`hole`, T`放置到 hole 工位`],
          [T`2`, T`缺口`, T`chip`, T`放置到 chip 工位`],
          [T`3`, T`划痕`, T`scratch`, T`放置到 scratch 工位`],
          [T`4`, T`污渍`, T`stain`, T`放置到 stain 工位`],
          [T`0`, T`未识别 / 失败`, T`unknown`, T`报警停机（valid=0）`],
        ],
        widths: [1500, 1500, 1500, 3500],
      },
      {
        type: "text",
        text: T`注意：视觉端已完成类别映射（YOLO 输出 0-based 索引经 +1 偏移后变为 1~4），PLC 端直接使用 HR[110] 的值即可，无需额外转换。`,
      },
    ],
  },
  {
    title: T`六、异常处理约定`,
    level: 1,
    page: 7,
    content: [
      {
        type: "text",
        text: T`双方在通讯异常或视觉检测失败时的处理约定如下：`,
      },
      {
        type: "table",
        headers: [T`场景`, T`视觉端行为`, T`PLC 端应处理`],
        rows: [
          [T`正常检测`, T`valid=1, seq=N, class=1~4`, T`读取结果，执行搬运`],
          [T`相机故障`, T`valid=0, class=0`, T`报警停机`],
          [T`推理超时`, T`valid=0, class=0`, T`报警停机`],
          [T`低置信度`, T`valid=0, class=0`, T`报警停机`],
          [T`Modbus 断开`, T`持续重连，不上报`, T`等待 N 秒无响应则报警`],
        ],
        widths: [2200, 3000, 3000],
      },
      {
        type: "text",
        text: T`设计原则：宁可报错停机，不可放错物料。当 valid=0 时，PLC 应立即停止当前周期并触发报警，等待人工干预或复位信号。`,
      },
    ],
  },
  {
    title: T`七、附录`,
    level: 1,
    page: 8,
    content: [
      { type: "h2", text: T`7.1 视觉端配置片段（config.yaml）` },
      {
        type: "text",
        text: T`以下配置片段供 PLC 工程师参考，了解视觉端的寄存器映射设定：`,
      },
      {
        type: "code",
        text: T`plc:
  protocol: "modbus_tcp"
  modbus_tcp:
    host: "192.168.3.99"
    port: 502
    unit_id: 1
    timeout_s: 2.0
    registers:
      trigger: 100
      defect_class: 110
      confidence: 111
      valid: 112
      seq: 113`,
      },
      { type: "h2", text: T`7.2 调试工具` },
      {
        type: "list",
        items: [
          T`python -m src.dummy_modbus_server — 本地 Modbus TCP 模拟服务器（端口 502）`,
          T`python -m test.test_plc_comm — 通讯连通性测试脚本`,
          T`Modbus Poll / Modbus Slave — 第三方 GUI 调试工具`,
        ],
      },
      { type: "h2", text: T`7.3 联系方式` },
      {
        type: "text",
        text: T`如对接口规范有疑问或需要调整寄存器地址、协议参数，请联系视觉端负责人协商修改 config.yaml 配置。`,
      },
    ],
  },
];

// ===== 渲染文档 =====
const children = [];

// 封面
children.push(
  para(run(docTitle, { bold: true, size: 44, color: palette.dark }), {
    alignment: AlignmentType.CENTER,
    spacing: { before: 2400, after: 400 },
  })
);
children.push(
  para(run(docSubtitle, { size: 28, color: palette.primary }), {
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
  })
);
children.push(
  para(run(docVersion, { size: 22, color: palette.light }), {
    alignment: AlignmentType.CENTER,
    spacing: { after: 1200 },
  })
);

// 分隔线（空段落模拟）
children.push(para(run(""), { spacing: { after: 400 } }));

// 目录标题
children.push(heading(T`目录`, 1));
children.push(
  para(run(T`右键目录，选择"更新域"刷新页码。`, { italics: true, color: palette.light, size: 20 }))
);
children.push(
  toc(
    sections.map(({ title: entryTitle, level, page }) => ({
      title: entryTitle,
      level,
      page,
    }))
  )
);

// 渲染各章节
for (const section of sections) {
  children.push(heading(section.title, section.level));
  for (const block of section.content) {
    switch (block.type) {
      case "text":
        children.push(bodyPara(block.text));
        break;
      case "h2":
        children.push(heading(block.text, 2));
        break;
      case "h3":
        children.push(heading(block.text, 3));
        break;
      case "table":
        children.push(makeTable(block.headers, block.rows, block.widths));
        children.push(para(run(""), { spacing: { after: 120 } }));
        break;
      case "list":
        for (const item of block.items) {
          children.push(
            para(run("• " + item), {
              indent: { left: convertInchesToTwip(0.3) },
              spacing: { after: 80 },
            })
          );
        }
        break;
      case "code":
        children.push(
          para(run(block.text, { font: { ...font, ascii: "Consolas", eastAsia: "SimSun" }, size: 20 }), {
            shading: { type: ShadingType.CLEAR, fill: "F5F5F5" },
            spacing: { before: 120, after: 120 },
            indent: { left: convertInchesToTwip(0.3) },
          })
        );
        break;
    }
  }
}

const doc = new Document({
  features: { updateFields: true },
  sections: [
    {
      properties: {
        page: {
          margin: {
            top: 1440,
            bottom: 1440,
            left: 1440,
            right: 1440,
          },
        },
      },
      headers: {
        default: new Header({
          children: [
            para(run(T`视觉系统 PLC 接口规范文档`, { bold: true, color: palette.primary, size: 20 }), {
              alignment: AlignmentType.CENTER,
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            para(
              new TextRun({ children: [PageNumber.CURRENT], size: 20 }),
              { alignment: AlignmentType.CENTER }
            ),
          ],
        }),
      },
      children,
    },
  ],
});

const buffer = await Packer.toBuffer(doc);
fs.writeFileSync(outputPath, buffer);
