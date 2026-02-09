import '@mantine/core/styles.css'
import './App.css'
import { MantineProvider, createTheme, Container, Title, Text, Textarea, Button, Group, Stack, Paper, Slider, Select, Badge, ActionIcon, Tooltip, Box, SimpleGrid, Code, CopyButton, Loader, Switch, NumberInput, Tabs, Divider, Center, Skeleton, Transition } from '@mantine/core'
import { useState, useEffect } from 'react'
import { Sparkles, Settings, Copy, Check, Image as ImageIcon, Terminal, Cpu, Palette, Wand2, Moon, Route, Brain, Home, Cat, Star, Mountain, TreePine, Heart, Lightbulb, ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const theme = createTheme({
  primaryColor: 'blue',
  fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
  defaultRadius: 'md',
  colors: {
    dark: [
      '#C1C2C5', '#A6A7AB', '#909296', '#5C5F66',
      '#373A40', '#2C2E33', '#25262B', '#1A1B1E',
      '#141517', '#101113'
    ],
  },
})

const EXAMPLES = [
  { icon: Home, label: 'House' },
  { icon: Cat, label: 'Cat on chair' },
  { icon: Star, label: 'Stars & moon' },
  { icon: Mountain, label: 'Mountain' },
  { icon: TreePine, label: 'Tree' },
  { icon: Heart, label: 'Heart' },
]

const IMAGE_MODELS = [
  { value: 'flux-hf', label: 'FLUX.1 Schnell (HuggingFace) - Best Quality' },
  { value: 'pollinations-flux', label: 'Pollinations FLUX - Free Fallback' },
  { value: 'pollinations-turbo', label: 'Pollinations Turbo - Fast' },
]

const ASCII_MODELS = [
  { value: 'resnet18', label: 'ResNet18 - Best' },
  { value: 'vit', label: 'ViT - Experimental' },
  { value: 'legacy', label: 'ResNet18 - Legacy' },
]

const RENDER_MODES = [
  { value: 'auto', label: 'AI Auto-Select (Best Quality)' },
  { value: 'ssim', label: 'Deep Structure (SSIM)' },
  { value: 'portrait', label: 'Portrait (Gradient)' },
  { value: 'cnn', label: 'Standard (CNN)' },
  { value: 'neat', label: 'Neat (Gradient)' },
  { value: 'standard', label: 'Standard (Gradient)' },
  { value: 'high', label: 'High (Gradient)' },
  { value: 'ultra', label: 'Ultra (Gradient)' },
]

function App() {
  const [prompt, setPrompt] = useState('')

  // Generation Settings
  const [width, setWidth] = useState(125)
  const [renderMode, setRenderMode] = useState('auto')
  const [imageModel, setImageModel] = useState('flux-hf')
  const [asciiModel, setAsciiModel] = useState('resnet18')

  // Advanced Options
  const [seed, setSeed] = useState(42)
  const [darkModeInvert, setDarkModeInvert] = useState(false)
  const [autoRouting, setAutoRouting] = useState(true)
  const [semanticPalette, setSemanticPalette] = useState(true)
  const [neuralMapper, setNeuralMapper] = useState(true)
  const [customToken, setCustomToken] = useState('')
  const [useCustomToken, setUseCustomToken] = useState(false)

  // State
  const [loading, setLoading] = useState(false)
  const [ascii, setAscii] = useState('')
  const [image, setImage] = useState('')
  const [logs, setLogs] = useState([])
  const [asciiFontSize, setAsciiFontSize] = useState(6)
  const [imageLoaded, setImageLoaded] = useState(false)

  // Reset image loaded state when image changes
  useEffect(() => {
    if (image) setImageLoaded(false)
  }, [image])

  const addLog = (msg, type = 'info') => {
    const now = new Date()
    const timestamp = now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
    setLogs(prev => [...prev.slice(-8), { msg, type, id: Date.now(), timestamp }])
  }

  const handleGenerate = async () => {
    if (!prompt.trim()) return
    setLoading(true)
    setLogs([])
    setAscii('')
    setImage('')

    addLog('Starting generation pipeline...')
    addLog(`Prompt: "${prompt}"`)
    addLog(`Mode: ${RENDER_MODES.find(m => m.value === renderMode)?.label}`)

    try {
      const res = await fetch(`${API_URL}/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          width,
          quality: renderMode,
          image_model: imageModel,
          ascii_model: asciiModel,
          seed,
          invert: darkModeInvert,
          auto_routing: autoRouting,
          semantic_palette: semanticPalette,
          neural_mapper: neuralMapper,
          custom_token: useCustomToken ? customToken : null
        })
      })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = await res.json()
      setAscii(data.ascii || '')
      setImage(data.image_url || '')
      addLog('Generation complete!', 'success')
    } catch (e) {
      addLog(`${e.message}`, 'error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <Box className="app-container" style={{ minHeight: '100vh', background: 'var(--mantine-color-dark-8)' }}>
        <Container size="xl" py="xl">
          {/* Header */}
          <Box mb={50} pt="md" className="header-section">
            <Group justify="space-between" align="flex-start" wrap="wrap">
              <Box>
                <Text size="xs" tt="uppercase" fw={700} c="blue.4" mb={8} style={{ letterSpacing: '0.15em' }} className="header-label">
                  Text-to-Art Pipeline
                </Text>
                <Title
                  order={1}
                  className="header-title"
                  style={{
                    fontSize: 'clamp(2.5rem, 5vw, 3.5rem)',
                    fontWeight: 800,
                    letterSpacing: '-0.03em',
                    lineHeight: 1.1,
                    marginBottom: '0.5rem'
                  }}
                >
                  ASCII Art
                  <Text
                    component="span"
                    inherit
                    className="gradient-text"
                  >
                    Generator
                  </Text>
                </Title>
                <Text c="dimmed" size="lg" maw={500} fw={400}>
                  Turn your ideas into stunning character-based art using the power of FLUX.1 and neural networks.
                </Text>
              </Box>
              <Badge
                variant="light"
                size="md"
                radius="sm"
                color="gray"
                className="version-badge"
              >
                v2.0 • Production Build
              </Badge>
            </Group>
            {/* Animated gradient line */}
            <Box className="gradient-line" mt="lg" />
          </Box>

          <SimpleGrid cols={{ base: 1, lg: 2 }} spacing={40}>
            {/* Left Column - Input & Settings */}
            <Stack gap="xl">
              {/* Creator Section */}
              <Box>
                <Group gap="xs" mb="sm" className="section-header">
                  <Box className="section-icon">
                    <Sparkles size={16} />
                  </Box>
                  <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                    Creator Studio
                  </Text>
                </Group>

                <Paper p="xl" radius="lg" withBorder bg="var(--mantine-color-dark-7)" className="glass-card">
                  <Stack gap="md">
                    <Textarea
                      size="lg"
                      label={<Text fw={600} mb={4}>Prompt</Text>}
                      description="Describe what you want to create - be specific about style, colors, and composition"
                      placeholder="A cyberpunk city street at night with neon rain..."
                      minRows={3}
                      value={prompt}
                      onChange={(e) => setPrompt(e.currentTarget.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && !e.shiftKey) {
                          e.preventDefault()
                          handleGenerate()
                        }
                      }}
                      styles={{ 
                        input: { 
                          fontSize: '1.1rem',
                          transition: 'border-color 0.2s, box-shadow 0.2s',
                        } 
                      }}
                      classNames={{ input: 'prompt-input' }}
                    />

                    <Box>
                      <Group gap={6} align="center" mb="sm">
                        <Lightbulb size={14} color="var(--mantine-color-yellow-5)" />
                        <Text size="xs" fw={600} c="dimmed">TRY AN EXAMPLE</Text>
                      </Group>
                      <Group gap={8} wrap="wrap">
                        {EXAMPLES.map(ex => {
                          const IconComponent = ex.icon
                          return (
                            <Button
                              key={ex.label}
                              variant="subtle"
                              color="gray"
                              size="xs"
                              radius="xl"
                              leftSection={<IconComponent size={14} />}
                              onClick={() => setPrompt(ex.label)}
                              className="example-chip"
                            >
                              {ex.label}
                            </Button>
                          )
                        })}
                      </Group>
                    </Box>
                  </Stack>
                </Paper>
              </Box>

              {/* Configuration Section */}
              <Box>
                <Group gap="xs" mb="sm" className="section-header">
                  <Box className="section-icon">
                    <Settings size={16} />
                  </Box>
                  <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                    Configuration
                  </Text>
                </Group>

                <Paper radius="lg" withBorder overflow="hidden" className="glass-card config-panel">
                  <Tabs defaultValue="basic" variant="pills" radius="md" p="sm">
                    <Tabs.List style={{ border: 'none' }} className="config-tabs">
                      <Tabs.Tab value="basic" leftSection={<Settings size={14} />} fw={600} className="config-tab">
                        Basic
                      </Tabs.Tab>
                      <Tabs.Tab value="models" leftSection={<Cpu size={14} />} fw={600} className="config-tab">
                        Models
                      </Tabs.Tab>
                      <Tabs.Tab value="advanced" leftSection={<Wand2 size={14} />} fw={600} className="config-tab">
                        Advanced
                      </Tabs.Tab>
                    </Tabs.List>

                    <Divider my="sm" color="white" opacity={0.06} />

                    <Tabs.Panel value="basic" p="xs">
                      <Stack gap="lg">
                        <Box>
                          <Group justify="space-between" mb="xs">
                            <Text size="sm" fw={600}>Output Width</Text>
                            <Badge variant="light" color="blue">{width} chars</Badge>
                          </Group>
                          <Slider
                            size="lg"
                            min={30}
                            max={150}
                            step={5}
                            value={width}
                            onChange={setWidth}
                            marks={[
                              { value: 30, label: '30' },
                              { value: 90, label: '90' },
                              { value: 150, label: '150' },
                            ]}
                          />
                        </Box>
                        <Select
                          label={<Text size="sm" fw={600} mb={4}>Render Mode</Text>}
                          description="Algorithm for converting image to ASCII"
                          data={RENDER_MODES}
                          value={renderMode}
                          onChange={setRenderMode}
                          size="md"
                        />
                      </Stack>
                    </Tabs.Panel>

                    <Tabs.Panel value="models" p="xs">
                      <Stack gap="md">
                        <Select
                          label={<Text size="sm" fw={600} mb={4}>Image Generation Model</Text>}
                          data={IMAGE_MODELS}
                          value={imageModel}
                          onChange={setImageModel}
                        />
                        <Select
                          label={<Text size="sm" fw={600} mb={4}>ASCII Mapping Model</Text>}
                          data={ASCII_MODELS}
                          value={asciiModel}
                          onChange={setAsciiModel}
                        />
                        <Paper p="sm" withBorder bg="dark.8" className="option-card">
                          <Switch
                            label={<Text size="sm" fw={500}>Enable Neural Mapper</Text>}
                            description="AI-enhanced character selection"
                            checked={neuralMapper}
                            onChange={(e) => setNeuralMapper(e.currentTarget.checked)}
                          />
                        </Paper>
                        <Box>
                          <Switch
                            label={<Text size="sm" fw={500}>Use Custom HuggingFace Token</Text>}
                            checked={useCustomToken}
                            onChange={(e) => setUseCustomToken(e.currentTarget.checked)}
                            mb="xs"
                          />
                          {useCustomToken && (
                            <Textarea
                              placeholder="hf_..."
                              value={customToken}
                              onChange={(e) => setCustomToken(e.currentTarget.value)}
                            />
                          )}
                        </Box>
                      </Stack>
                    </Tabs.Panel>

                    <Tabs.Panel value="advanced" p="xs">
                      <Stack gap="md">
                        <NumberInput
                          label={<Text size="sm" fw={600} mb={4}>Seed</Text>}
                          value={seed}
                          onChange={setSeed}
                          min={0}
                          max={999999}
                        />
                        <SimpleGrid cols={2}>
                          <Paper p="sm" withBorder bg="dark.8" className="option-card">
                            <Switch
                              label={<Group gap={6}><Moon size={14} /><Text size="sm" fw={500}>Dark Mode</Text></Group>}
                              checked={darkModeInvert}
                              onChange={(e) => setDarkModeInvert(e.currentTarget.checked)}
                            />
                          </Paper>
                          <Paper p="sm" withBorder bg="dark.8" className="option-card">
                            <Switch
                              label={<Group gap={6}><Route size={14} /><Text size="sm" fw={500}>Auto-Routing</Text></Group>}
                              checked={autoRouting}
                              onChange={(e) => setAutoRouting(e.currentTarget.checked)}
                            />
                          </Paper>
                          <Paper p="sm" withBorder bg="dark.8" style={{ gridColumn: 'span 2' }} className="option-card">
                            <Switch
                              label={<Group gap={6}><Palette size={14} /><Text size="sm" fw={500}>Semantic Palette</Text></Group>}
                              description="Use subject-aware characters"
                              checked={semanticPalette}
                              onChange={(e) => setSemanticPalette(e.currentTarget.checked)}
                            />
                          </Paper>
                        </SimpleGrid>
                      </Stack>
                    </Tabs.Panel>
                  </Tabs>
                </Paper>
              </Box>

              {/* Generate Button */}
              <Button
                size="xl"
                fullWidth
                radius="md"
                gradient={{ from: 'blue', to: 'cyan', deg: 90 }}
                variant="gradient"
                leftSection={loading ? <Loader size={20} color="white" /> : <Sparkles size={20} />}
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                className={`generate-btn ${!loading && prompt.trim() ? 'ready-pulse' : ''}`}
              >
                {loading ? 'Generating Art...' : 'Generate ASCII Art'}
              </Button>

              {/* Process Log */}
              <Transition mounted={logs.length > 0} transition="slide-up" duration={300}>
                {(styles) => (
                  <Paper p="sm" withBorder bg="dark.8" className="process-log" style={styles}>
                    <Group justify="space-between" mb="xs">
                      <Text size="xs" fw={700} c="dimmed" tt="uppercase" style={{ letterSpacing: '0.05em' }}>
                        Thinking Process
                      </Text>
                      <Brain size={14} color="var(--mantine-color-dimmed)" />
                    </Group>
                    <Code block style={{ background: 'transparent', fontSize: 11, lineHeight: 1.6 }}>
                      {logs.map(l => (
                        <div key={l.id} className={`log-entry log-${l.type}`}>
                          <span className="log-timestamp">[{l.timestamp}]</span>
                          <span className="log-arrow">›</span>
                          <span>{l.msg}</span>
                        </div>
                      ))}
                    </Code>
                  </Paper>
                )}
              </Transition>
            </Stack>

            {/* Right Column - Output */}
            <Stack gap="xl">
              <Box>
                <Group gap="xs" mb="sm" className="section-header">
                  <Box className="section-icon">
                    <ImageIcon size={16} />
                  </Box>
                  <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                    Visual Output
                  </Text>
                </Group>

                <Paper p="xs" radius="lg" withBorder bg="var(--mantine-color-dark-8)" className="glass-card output-card">
                  <Box
                    style={{
                      aspectRatio: '1',
                      background: 'var(--mantine-color-dark-7)',
                      borderRadius: 'var(--mantine-radius-md)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      overflow: 'hidden',
                      position: 'relative'
                    }}
                  >
                    {loading && !image ? (
                      <Stack align="center" gap="md">
                        <Skeleton height={200} width={200} radius="md" animate />
                        <Skeleton height={16} width={150} radius="sm" animate />
                      </Stack>
                    ) : image ? (
                      <>
                        {!imageLoaded && (
                          <Box style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <Loader size="lg" />
                          </Box>
                        )}
                        <img 
                          src={image} 
                          alt="Generated" 
                          style={{ 
                            maxWidth: '100%', 
                            maxHeight: '100%', 
                            objectFit: 'contain',
                            opacity: imageLoaded ? 1 : 0,
                            transition: 'opacity 0.3s ease'
                          }} 
                          onLoad={() => setImageLoaded(true)}
                        />
                      </>
                    ) : (
                      <Stack align="center" gap="xs" className="empty-state">
                        <Box className="empty-icon">
                          <ImageIcon size={48} strokeWidth={1} />
                        </Box>
                        <Text size="sm" c="dimmed">Image will appear here</Text>
                        <Text size="xs" c="dimmed" style={{ opacity: 0.5 }}>Enter a prompt and click generate</Text>
                      </Stack>
                    )}
                  </Box>
                </Paper>
              </Box>

              <Box style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
                <Group justify="space-between" mb="sm">
                  <Group gap="xs" className="section-header">
                    <Box className="section-icon">
                      <Terminal size={16} />
                    </Box>
                    <Text tt="uppercase" size="xs" fw={700} c="dimmed" style={{ letterSpacing: '0.1em' }}>
                      ASCII Result
                    </Text>
                  </Group>
                  <Group gap="xs">
                    {ascii && (
                      <>
                        <Tooltip label="Zoom out">
                          <ActionIcon 
                            variant="subtle" 
                            color="gray" 
                            radius="md" 
                            onClick={() => setAsciiFontSize(s => Math.max(3, s - 1))}
                          >
                            <ZoomOut size={16} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Reset zoom">
                          <ActionIcon 
                            variant="subtle" 
                            color="gray" 
                            radius="md" 
                            onClick={() => setAsciiFontSize(6)}
                          >
                            <RotateCcw size={16} />
                          </ActionIcon>
                        </Tooltip>
                        <Tooltip label="Zoom in">
                          <ActionIcon 
                            variant="subtle" 
                            color="gray" 
                            radius="md" 
                            onClick={() => setAsciiFontSize(s => Math.min(12, s + 1))}
                          >
                            <ZoomIn size={16} />
                          </ActionIcon>
                        </Tooltip>
                        <Box w={1} h={16} bg="dark.5" />
                        <CopyButton value={ascii}>
                          {({ copied, copy }) => (
                            <Tooltip label={copied ? 'Copied!' : 'Copy raw text'}>
                              <ActionIcon variant="light" onClick={copy} color={copied ? 'teal' : 'gray'} radius="md">
                                {copied ? <Check size={16} /> : <Copy size={16} />}
                              </ActionIcon>
                            </Tooltip>
                          )}
                        </CopyButton>
                      </>
                    )}
                  </Group>
                </Group>

                <Paper flex={1} radius="lg" withBorder overflow="hidden" style={{ display: 'flex', flexDirection: 'column' }} className="glass-card ascii-output">
                  <Box
                    p="md"
                    style={{
                      background: '#0d1117',
                      flex: 1,
                      minHeight: 400,
                      overflow: 'auto',
                      position: 'relative'
                    }}
                  >
                    {loading && !ascii ? (
                      <Box py="xl">
                        <Stack gap="xs" align="center">
                          <Skeleton height={8} width="80%" radius="sm" animate />
                          <Skeleton height={8} width="90%" radius="sm" animate />
                          <Skeleton height={8} width="75%" radius="sm" animate />
                          <Skeleton height={8} width="85%" radius="sm" animate />
                          <Skeleton height={8} width="70%" radius="sm" animate />
                        </Stack>
                      </Box>
                    ) : ascii ? (
                      <Transition mounted={!!ascii} transition="fade" duration={400}>
                        {(styles) => (
                          <pre style={{
                            ...styles,
                            fontFamily: 'JetBrains Mono, Fira Code, monospace',
                            fontSize: asciiFontSize,
                            lineHeight: 1,
                            color: '#58a6ff',
                            margin: 0,
                            whiteSpace: 'pre'
                          }}>
                            {ascii}
                          </pre>
                        )}
                      </Transition>
                    ) : (
                      <Center h="100%">
                        <Stack align="center" gap="xs" className="empty-state">
                          <Box className="empty-icon">
                            <Terminal size={48} strokeWidth={1} />
                          </Box>
                          <Text size="sm" c="dimmed">ASCII art will appear here</Text>
                          <Text size="xs" c="dimmed" style={{ opacity: 0.5 }}>Your generated art in characters</Text>
                        </Stack>
                      </Center>
                    )}
                  </Box>
                  {ascii && (
                    <Box p="xs" bg="dark.8" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }} className="ascii-stats">
                      <Group justify="space-between">
                        <Group gap="xs">
                          <Badge size="xs" variant="dot" color="blue">READY</Badge>
                        </Group>
                        <Group gap="xs">
                          <Text size="xs" c="dimmed">{width} chars wide</Text>
                          <Text size="xs" c="dimmed">•</Text>
                          <Text size="xs" c="dimmed">{ascii.split('\n').length} lines</Text>
                          <Text size="xs" c="dimmed">•</Text>
                          <Text size="xs" c="dimmed">{asciiFontSize}px font</Text>
                        </Group>
                      </Group>
                    </Box>
                  )}
                </Paper>
              </Box>
            </Stack>
          </SimpleGrid>
        </Container>
      </Box>
    </MantineProvider>
  )
}

export default App
