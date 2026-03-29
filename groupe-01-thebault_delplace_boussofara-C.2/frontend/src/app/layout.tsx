import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "BrightVest RAG",
  description: "Financial AI Chat Assistant powered by RAG",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="font-sans bg-slate-900 text-slate-100 antialiased">
        {children}
      </body>
    </html>
  );
}
