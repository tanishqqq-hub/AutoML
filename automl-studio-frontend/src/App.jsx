/**
 * App.jsx — Router with all pages and 404 fallback.
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout   from './components/Layout'
import Upload   from './pages/Upload'
import Train    from './pages/Train'
import Results  from './pages/Results'
import Predict  from './pages/Predict'
import NotFound from './pages/NotFound'

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/"        element={<Upload   />} />
          <Route path="/train"   element={<Train    />} />
          <Route path="/results" element={<Results  />} />
          <Route path="/predict" element={<Predict  />} />
          <Route path="*"        element={<NotFound />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  )
}