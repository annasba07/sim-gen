import React from 'react'
import { render, screen } from '@testing-library/react'

describe('Simple Test', () => {
  it('should pass a basic assertion', () => {
    expect(true).toBe(true)
  })

  it('should render a simple component', () => {
    const TestComponent = () => <div>Test Component</div>
    render(<TestComponent />)
    expect(screen.getByText('Test Component')).toBeInTheDocument()
  })
})