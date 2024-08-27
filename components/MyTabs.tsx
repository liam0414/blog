import React, { useState, ReactNode, useEffect } from "react";
import { Tabs, Tab, Box, Typography } from "@mui/material";

interface TabPanelProps {
  children?: ReactNode;
  index: number;
  value: number;
}

const TabPanel = (props: TabPanelProps) => {
  const { children, value, index, ...other } = props;
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`simple-tabpanel-${index}`}
      aria-labelledby={`simple-tab-${index}`}
      {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
};

interface MyTabsProps {
  children: ReactNode[];
}

const MyTabs: React.FC<MyTabsProps> = ({ children }) => {
  const [value, setValue] = useState(0);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  const handleChange = (event: React.SyntheticEvent, newValue: number) => {
    setValue(newValue);
  };

  const tabs = React.Children.toArray(children).filter(
    (child) => React.isValidElement(child) && child.type === Tab
  );

  const tabPanels = React.Children.toArray(children).filter(
    (child) => React.isValidElement(child) && child.type === TabPanel
  );

  if (!isClient) {
    return null;
  }

  return (
    <Box sx={{ width: "100%" }}>
      <Box sx={{ borderBottom: 1, borderColor: "divider" }}>
        <Tabs value={value} onChange={handleChange} aria-label="basic tabs example">
          {tabs}
        </Tabs>
      </Box>
      {React.Children.map(tabPanels, (panel, index) =>
        React.cloneElement(panel as React.ReactElement<any>, { value, index })
      )}
    </Box>
  );
};

export { MyTabs, TabPanel, Tab };
