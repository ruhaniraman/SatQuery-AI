import { useEffect, useState } from 'react';

// width / height of the image at `url` once it has loaded (null until then, and for a broken image), so
// the viewer can work out where an object-contain picture really sits.
export function useImageAspect(url) {
  const [aspect, setAspect] = useState({ url: null, value: null });
  useEffect(() => {
    if (!url) return undefined;
    let live = true;
    const image = new Image();
    image.onload = () => { if (live && image.naturalHeight > 0) setAspect({ url, value: image.naturalWidth / image.naturalHeight }); };
    image.src = url;
    return () => { live = false; };
  }, [url]);
  return aspect.url === url ? aspect.value : null;
}
