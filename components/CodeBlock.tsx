import React from "react";
import { Box } from "@mui/material";

interface CodeBlockProps {
  children: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ children }) => {
  return (
    <Box
      component="pre"
      sx={{
        backgroundColor: "#f5f5f5",
        borderRadius: "4px",
        padding: "16px",
        overflowX: "auto",
        "& code": {
          fontFamily: 'Consolas, Monaco, "Andale Mono", "Ubuntu Mono", monospace',
          fontSize: "14px",
          lineHeight: 1.5,
        },
      }}>
      <code>{children}</code>
    </Box>
  );
};

export default CodeBlock;
