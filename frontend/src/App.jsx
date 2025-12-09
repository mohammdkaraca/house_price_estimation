import React, { useState, useEffect } from 'react';

// --- Icons ---
const DashboardIcon = () => <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>;
const LightningIcon = () => <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>;

export default function App() {
  // --- Form Configuration (Matches your CSV Columns) ---
  const [turkishCities, setTurkishCities] = useState([]);
  const [commonOptions, setCommonOptions] = useState({
    Bulunduğu_Kat: [],
    Isıtma_Tipi: [],
    Binanın_Yaşı: [],
    Eşya_Durumu: [],
    Kullanım_Durumu: [],
    Yatırıma_Uygunluk: [],
    Tapu_Durumu: [],
    Takas: []
  });

  useEffect(() => {
    // Fetch Turkish cities from backend
    fetch('http://127.0.0.1:8000/cities')
      .then(res => res.json())
      .then(data => setTurkishCities(data.cities))
      .catch(err => console.error('Failed to fetch cities:', err));
  }, []);

  const formFields = [
    { key: 'Net_Metrekare', label: 'Net Area (m²)', type: 'number', placeholder: 'e.g. 120' },
    { key: 'Brüt_Metrekare', label: 'Gross Area (m²)', type: 'number', placeholder: 'e.g. 150' },
    { key: 'Oda_Sayısı', label: 'Room Count', type: 'number', placeholder: 'e.g. 3' },
    { key: 'Banyo_Sayısı', label: 'Bathroom Count', type: 'number', placeholder: 'e.g. 1' },
    { key: 'Binanın_Kat_Sayısı', label: 'Total Floors in Building', type: 'number', placeholder: 'e.g. 10' },
    { 
      key: 'Şehir', 
      label: 'City', 
      type: 'select', 
      options: turkishCities.length > 0 ? turkishCities : ['Adana', 'Istanbul', 'Ankara', 'Izmir', 'Konya'],
      placeholder: 'Select a city'
    },
    { 
      key: 'Bulunduğu_Kat', 
      label: 'Floor Level', 
      type: 'select', 
      options: ['4.Kat', '3.Kat', '6.Kat', 'Düz Giriş (Zemin)', '12.Kat', '2.Kat', 'nan', '8.Kat',
                '5.Kat', '14.Kat', '16.Kat', '1.Kat', '17.Kat', '9.Kat', 'Yüksek Giriş', '7.Kat',
                '11.Kat', 'Müstakil', 'Bahçe Katı', '10.Kat', '15.Kat', 'Bahçe Dublex',
                '13.Kat', 'Kot 3 (-3).Kat', 'Villa Tipi', '18.Kat', 'Çatı Dubleks', '21.Kat',
                'Kot 2 (-2).Kat', 'Bodrum Kat', 'Çatı Katı', '26.Kat', 'Kot 1 (-1).Kat',
                'Kot 4 (-4).Kat', '40+.Kat', '19.Kat', '30.Kat', '22.Kat'],
      placeholder: 'Select floor'
    },
    { 
      key: 'Isıtma_Tipi', 
      label: 'Heating Type', 
      type: 'select', 
      options: ['Bilinmiyor', 'Kombi Doğalgaz', 'Klimalı', 'Merkezi (Pay Ölçer)', 'Sobalı', 
               'Isıtma Yok', 'Doğalgaz Sobalı', 'Merkezi Doğalgaz', 'Yerden Isıtma', 
               'Güneş Enerjisi', 'Kat Kaloriferi', 'Diğer'],
      placeholder: 'Select heating type'
    },
    { 
      key: 'Binanın_Yaşı', 
      label: 'Building Age', 
      type: 'select', 
      options: ['Bilinmiyor', '0 (Yeni)', '1', '2', '3', '4', '5-10', '11-15', '16-20', '21 Ve Üzeri'],
      placeholder: 'Select building age'
    },
    { 
      key: 'Eşya_Durumu', 
      label: 'Furnished Status', 
      type: 'select', 
      options: ['Bilinmiyor', 'Eşyalı', 'Boş'],
      placeholder: 'Select furnished status'
    },
    { 
      key: 'Kullanım_Durumu', 
      label: 'Usage Status', 
      type: 'select', 
      options: ['Bilinmiyor', 'Boş', 'Kiracı Oturuyor', 'Mülk Sahibi Oturuyor'],
      placeholder: 'Select usage status'
    },
    { 
      key: 'Yatırıma_Uygunluk', 
      label: 'Investment Suitability', 
      type: 'select', 
      options: ['Bilinmiyor', 'Uygun', 'Bilinmiyor'],
      placeholder: 'Select investment suitability'
    },
    { 
      key: 'Tapu_Durumu', 
      label: 'Title Deed Status', 
      type: 'select', 
      options: ['Bilinmiyor', 'Kat Mülkiyeti', 'Kat İrtifakı', 'Müstakil Tapulu', 
               'Arsa Tapulu', 'Hisseli Tapu'],
      placeholder: 'Select title deed status'
    },
    { 
      key: 'Takas', 
      label: 'Swap Available', 
      type: 'select', 
      options: ['Bilinmiyor', 'Var', 'Yok'],
      placeholder: 'Select swap availability'
    },
  ];

  // --- State ---
  const [formData, setFormData] = useState({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // --- Handlers ---
  const handleInputChange = (key, value) => {
    setFormData(prev => ({ ...prev, [key]: value }));
  };

  async function submit() {
    setError(null);
    setResult(null);
    setLoading(true);

    // 1. Prepare Payload - convert empty strings to null
    const payload = {};
    formFields.forEach(field => {
      const rawValue = formData[field.key];
      
      if (rawValue === undefined || rawValue === '' || rawValue === null) {
        // For select fields, use 'Bilinmiyor' as default
        if (field.type === 'select') {
          payload[field.key] = "Bilinmiyor";
        } else {
          // For number fields, send null
          payload[field.key] = null;
        }
      } else {
        // Convert numbers properly
        if (field.type === 'number') {
          payload[field.key] = Number(rawValue);
        } else {
          payload[field.key] = rawValue;
        }
      }
    });

    console.log("Sending Payload:", payload);

    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Server error (${res.status}): ${errorText}`);
      }

      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
      console.error("Prediction error:", e);
    } finally {
      setLoading(false);
    }
  }

  // --- Styles ---
  const s = {
    page: { 
      minHeight: '100vh', 
      backgroundColor: '#0f172a', 
      color: '#cbd5e1', 
      fontFamily: '"Inter", sans-serif', 
      display: 'flex', 
      flexDirection: 'column' 
    },
    navbar: { 
      backgroundColor: '#1e293b', 
      borderBottom: '1px solid #334155', 
      padding: '16px 32px', 
      display: 'flex', 
      alignItems: 'center', 
      gap: '12px' 
    },
    brand: { 
      fontSize: '20px', 
      fontWeight: '700', 
      color: '#f8fafc' 
    },
    main: { 
      flex: 1, 
      padding: '32px', 
      maxWidth: '1400px', 
      margin: '0 auto', 
      width: '100%', 
      boxSizing: 'border-box', 
      display: 'grid', 
      gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', 
      gap: '32px', 
      alignItems: 'start' 
    },
    card: { 
      backgroundColor: '#1e293b', 
      borderRadius: '12px', 
      border: '1px solid #334155', 
      overflow: 'hidden', 
      boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.3)' 
    },
    cardHeader: { 
      padding: '20px 24px', 
      borderBottom: '1px solid #334155', 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center' 
    },
    cardTitle: { 
      fontSize: '16px', 
      fontWeight: '600', 
      color: '#f1f5f9', 
      display: 'flex', 
      alignItems: 'center', 
      gap: '8px' 
    },
    cardBody: { 
      padding: '24px' 
    },
    gridContainer: { 
      display: 'grid', 
      gridTemplateColumns: '1fr 1fr', 
      gap: '20px' 
    },
    inputGroup: { 
      display: 'flex', 
      flexDirection: 'column' 
    },
    label: { 
      fontSize: '12px', 
      fontWeight: '600', 
      color: '#94a3b8', 
      marginBottom: '6px', 
      textTransform: 'uppercase', 
      letterSpacing: '0.5px' 
    },
    input: { 
      backgroundColor: '#0f172a', 
      border: '1px solid #334155', 
      borderRadius: '6px', 
      padding: '10px', 
      color: '#f8fafc', 
      fontSize: '14px', 
      outline: 'none', 
      transition: 'border-color 0.2s',
      '&:focus': {
        borderColor: '#4f46e5'
      }
    },
    select: { 
      backgroundColor: '#0f172a', 
      border: '1px solid #334155', 
      borderRadius: '6px', 
      padding: '10px', 
      color: '#f8fafc', 
      fontSize: '14px', 
      outline: 'none',
      '&:focus': {
        borderColor: '#4f46e5'
      }
    },
    btnPrimary: { 
      width: '100%', 
      padding: '14px', 
      backgroundColor: '#4f46e5', 
      color: '#fff', 
      border: 'none', 
      borderRadius: '8px', 
      fontSize: '15px', 
      fontWeight: '600', 
      cursor: 'pointer', 
      marginTop: '24px', 
      transition: 'all 0.2s',
      '&:hover': {
        backgroundColor: '#4338ca'
      },
      '&:disabled': {
        opacity: 0.5,
        cursor: 'not-allowed'
      }
    },
    resultBox: { 
      textAlign: 'center', 
      padding: '30px', 
      backgroundColor: '#064e3b', 
      borderRadius: '8px', 
      border: '1px solid #059669', 
      color: '#fff' 
    },
    vizGrid: { 
      display: 'grid', 
      gridTemplateColumns: 'repeat(2, 1fr)', 
      gap: '16px', 
      marginTop: '16px' 
    },
    vizItem: { 
      backgroundColor: '#0f172a', 
      borderRadius: '8px', 
      padding: '8px', 
      border: '1px solid #334155' 
    },
    vizImg: { 
      width: '100%', 
      height: '140px', 
      objectFit: 'contain', 
      backgroundColor: '#1e293b', 
      borderRadius: '4px' 
    },
    errorBox: {
      marginTop: '20px', 
      padding: '12px', 
      background: '#450a0a', 
      border: '1px solid #7f1d1d', 
      color: '#fca5a5', 
      borderRadius: '6px', 
      fontSize: '13px'
    },
    placeholderText: {
      textAlign: 'center', 
      color: '#64748b', 
      padding: '40px'
    }
  };

  return (
    <div style={s.page}>
      <div style={s.navbar}>
        <DashboardIcon />
        <span style={s.brand}>RealEstate AI - Turkey</span>
      </div>

      <div style={s.main}>
        {/* LEFT: Input Form */}
        <div style={s.card}>
          <div style={s.cardHeader}>
            <div style={s.cardTitle}><LightningIcon /> Property Details</div>
            <button 
              onClick={() => setFormData({})} 
              style={{
                background: 'none', 
                border: 'none', 
                color: '#64748b', 
                cursor: 'pointer', 
                fontSize: '12px',
                textDecoration: 'underline'
              }}
            >
              Clear All
            </button>
          </div>
          <div style={s.cardBody}>
            <div style={s.gridContainer}>
              {formFields.map(field => (
                <div key={field.key} style={s.inputGroup}>
                  <label style={s.label}>{field.label}</label>
                  {field.type === 'select' ? (
                    <select 
                      style={s.select} 
                      value={formData[field.key] || ''}
                      onChange={e => handleInputChange(field.key, e.target.value)}
                    >
                      <option value="">{field.placeholder || 'Select...'}</option>
                      {field.options.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type={field.type}
                      style={s.input}
                      placeholder={field.placeholder}
                      value={formData[field.key] || ''}
                      onChange={e => handleInputChange(field.key, e.target.value)}
                      min={field.type === 'number' ? "0" : undefined}
                      step={field.type === 'number' ? "any" : undefined}
                    />
                  )}
                </div>
              ))}
            </div>

            <button 
              style={{
                ...s.btnPrimary, 
                opacity: loading ? 0.7 : 1
              }} 
              onClick={submit} 
              disabled={loading}
            >
              {loading ? 'Calculating...' : 'Predict Price'}
            </button>

            {error && (
              <div style={s.errorBox}>
                <strong>Error:</strong> {error}
              </div>
            )}
          </div>
        </div>

        {/* RIGHT: Results & Charts */}
        <div style={{display: 'flex', flexDirection: 'column', gap: '32px'}}>
          <div style={s.card}>
            <div style={s.cardHeader}>
              <div style={s.cardTitle}>Prediction Result</div>
            </div>
            <div style={s.cardBody}>
              {result ? (
                <div style={s.resultBox}>
                  <div style={{fontSize: '13px', opacity: 0.8, marginBottom: '8px'}}>
                    ESTIMATED PROPERTY VALUE
                  </div>
                  <div style={{fontSize: '36px', fontWeight: '700', marginBottom: '16px'}}>
                    ₺ {Number(result.predicted_price).toLocaleString('tr-TR')}
                  </div>
                  <div style={{fontSize: '12px', opacity: 0.7}}>
                    Based on {result.features_used || 'multiple'} features
                  </div>
                </div>
              ) : (
                <div style={s.placeholderText}>
                  Enter property details to see the price estimate.
                </div>
              )}
            </div>
          </div>

          
                
              </div>
            </div>
          </div>
  );
}