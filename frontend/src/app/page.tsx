import { redirect } from 'next/navigation'
import { DashboardLayout } from '@/components/layout/DashboardLayout'

export default function HomePage() {
  // In a real app, check authentication here
  // For now, redirect to dashboard
  redirect('/dashboard')
}