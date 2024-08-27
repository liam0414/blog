import React, { useEffect, useState, ReactNode } from "react";
import { Alert, AlertTitle, AlertProps } from "@mui/material";

interface MyAlertProps extends AlertProps {
  title: string;
  children: ReactNode;
}

const MyAlert: React.FC<MyAlertProps> = ({ severity, title, children, ...rest }) => {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <Alert 
      severity={severity} 
      sx={{ 
        margin: '8px 0',
        '& .MuiAlert-message': {
          width: '100%'
        }
      }} 
      {...rest}
    >
      <AlertTitle>{title}</AlertTitle>
      {children}
    </Alert>
  );
};

export default MyAlert;